package monitor

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	psnet "github.com/shirou/gopsutil/net"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
)

const (
	metricKeyNetworkOutKBPS     = "NetworkOutKBPS"
	metricKeyNetworkInKBPS      = "NetworkInKBPS"
	metricKeyIncomingPacketLoss = "PacketLossIn"
	metricKeyOutgoingPacketLoss = "PacketLossOut"

	bytesPerKilobyte = 1024.0
)

func getActivePhysicalNetworkInterfaces() ([]string, error) {
	allifaces, err := net.Interfaces()
	if err != nil {
		return nil, fmt.Errorf("failed to get network interfaces: %w", err)
	}

	var ifaces []string
	for _, iface := range allifaces {
		// skip if not up
		if iface.Flags&net.FlagUp == 0 {
			continue
		}

		// skip if we can't get the device's addresses
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}

		// skip if the device has no addresses
		if len(addrs) == 0 {
			continue
		}

		link, err := os.Readlink(filepath.Join("/sys/class/net", iface.Name))
		if err != nil {
			return nil, fmt.Errorf("failed to read sysfs info for %q: %w", iface.Name, err)
		}

		if strings.HasPrefix(link, "../../devices/virtual/") {
			continue
		}

		ifaces = append(ifaces, iface.Name)
	}

	if len(ifaces) == 0 {
		return nil, errors.New("no active network interfaces discovered")
	}

	return ifaces, nil
}

// returns a mapping of interface name to IOCountersStat obj for that interface
func getNetworkCounters(interfaceNames ...string) (map[string]psnet.IOCountersStat, error) {
	counters, err := psnet.IOCounters(true)
	if err != nil {
		return nil, fmt.Errorf("failed to get network counters: %w", err)
	}

	interfaceMap := make(map[string]struct{})
	for _, interfaceName := range interfaceNames {
		interfaceMap[interfaceName] = struct{}{}
	}

	countersMap := make(map[string]psnet.IOCountersStat)
	for _, counter := range counters {
		if _, ok := interfaceMap[counter.Name]; ok {
			countersMap[counter.Name] = counter
		}
	}

	if len(countersMap) != len(interfaceNames) {
		missingInterfaces := []string{}
		for _, name := range interfaceNames {
			_, ok := countersMap[name]
			if !ok {
				missingInterfaces = append(missingInterfaces, name)
			}
		}

		return nil, fmt.Errorf("Unable to fetch stats for the following interfaces %v", missingInterfaces)
	}

	return countersMap, nil
}

func calculateIOTransmissionRate(curCounters, prevCounters psnet.IOCountersStat, timeElapsed float64) (float64, float64) {
	kbOut := float64(curCounters.BytesSent-prevCounters.BytesSent) / bytesPerKilobyte
	kbIn := float64(curCounters.BytesRecv-prevCounters.BytesRecv) / bytesPerKilobyte

	return kbIn / timeElapsed, kbOut / timeElapsed
}

func calculateIOPacketLossPercentage(curCounters, prevCounters psnet.IOCountersStat) (float64, float64) {
	dropInDelta := float64(curCounters.Dropin - prevCounters.Dropin)
	totalInDelta := float64(curCounters.PacketsRecv - prevCounters.PacketsRecv)

	dropOutDelta := float64(curCounters.Dropout - prevCounters.Dropout)
	totalOutDelta := float64(curCounters.PacketsSent - prevCounters.PacketsSent)

	inLossPct := 0.0
	outLossPct := 0.0

	if dropInDelta != 0.0 && totalInDelta != 0.0 {
		inLossPct = dropInDelta / totalInDelta * 100.0
	}

	if dropOutDelta != 0.0 && totalOutDelta != 0.0 {
		outLossPct = dropOutDelta / totalOutDelta * 100.0
	}

	return inLossPct, outLossPct
}

// NetworkInterfaceStats monitors the following on a given network interface
// - input & output rate of network data transfer (in Kilobytes per second)
// - drop rate of incoming and outgoing packets (% dropped of total sent)
func NetworkInterfaceStats(ctx context.Context, interval time.Duration, interfaceNames ...string) error {
	prevCounters, err := getNetworkCounters(interfaceNames...)
	if err != nil {
		return fmt.Errorf("failed to measure net usage: %w", err)
	}
	lastCountTime := time.Now()

	for {
		time.Sleep(interval)
		curCounters, err := getNetworkCounters(interfaceNames...)
		if err != nil {
			return fmt.Errorf("failed to measure net usage: %w", err)
		}

		now := time.Now()
		elapsed := now.Sub(lastCountTime).Seconds()
		lastCountTime = now

		for _, interfaceName := range interfaceNames {

			inTransmissionRate, outTransmissionRate := calculateIOTransmissionRate(
				curCounters[interfaceName],
				prevCounters[interfaceName],
				elapsed,
			)

			inPacketLoss, outPacketLoss := calculateIOPacketLossPercentage(
				curCounters[interfaceName],
				prevCounters[interfaceName],
			)

			metricsctx.Publish(
				ctx,
				metricKeyNetworkInKBPS,
				time.Now(),
				metricsctx.UnitKilobytesSecond,
				inTransmissionRate,
				metricsctx.Dimensions{"InterfaceName": interfaceName},
			)

			metricsctx.Publish(
				ctx,
				metricKeyNetworkOutKBPS,
				time.Now(),
				metricsctx.UnitKilobytesSecond,
				outTransmissionRate,
				metricsctx.Dimensions{"InterfaceName": interfaceName},
			)

			metricsctx.Publish(
				ctx,
				metricKeyIncomingPacketLoss,
				now,
				metricsctx.UnitPercent,
				inPacketLoss,
				metricsctx.Dimensions{"InterfaceName": interfaceName},
			)

			metricsctx.Publish(
				ctx,
				metricKeyOutgoingPacketLoss,
				now,
				metricsctx.UnitPercent,
				outPacketLoss,
				metricsctx.Dimensions{"InterfaceName": interfaceName},
			)
		}

		prevCounters = curCounters
	}
}

// NetworkStats monitors the following on all active network interfaces
// - input & output rate of network data transfer (in Kilobytes per second)
// - drop rate of incoming and outgoing packets (% dropped of total sent)
func NetworkStats(ctx context.Context, interval time.Duration) error {
	interfaceNames, err := getActivePhysicalNetworkInterfaces()
	if err != nil {
		return fmt.Errorf("failed to determine physical network interfaces: %w", err)
	}

	return NetworkInterfaceStats(ctx, interval, interfaceNames...)
}
