package main

import (
	"context"
	"database/sql"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/cip8/autoname"
	"github.com/google/uuid"
	"github.com/lib/pq"
	_ "github.com/lib/pq"
	"gopkg.in/yaml.v3"

	"polygon_initial/app/polygon_initial"
)

// trunk-ignore-all(golangci-lint)
const (
	CSV_FILE                 = "<replace>"
	DEFAULT_EDGE_LIFECYCLE   = "provisioned"
	DEFAULT_CAMERA_LIFECYCLE = "onboarding"
	YAML_FILE_NAME           = "record-%d_output.yaml"
	DATABASE_HOST            = "<replace>"
	DATABASE_NAME            = "<replace>"
	DATABASE_USERNAME        = "<replace>"
	DATABASE_PASSWORD        = "<replace>"
	DATABASE_PORT            = "5432"
)

type processedRecord struct {
	Record       []string                        `yaml:"record"`
	Organization polygon_initial.ApiOrganization `yaml:"organization"`
	Zone         polygon_initial.Zone            `yaml:"zone"`
	Edge         polygon_initial.Edge            `yaml:"edge"`
	Cameras      []polygon_initial.Camera        `yaml:"cameras"`
}

func (pr *processedRecord) toYaml() ([]byte, error) {
	data, err := yaml.Marshal(&pr)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// maps all organizations to their names
// NOTE: takes an assumption that unique names are enforced within the database
func mapAllOrganizations(organizations []polygon_initial.ApiOrganization) map[string]polygon_initial.ApiOrganization {
	mappedOrginizations := make(map[string]polygon_initial.ApiOrganization)
	for _, value := range organizations {
		if _, ok := mappedOrginizations[value.Name]; ok { // if the key already exists
			panic("Duplicate organization Name")
		}
		mappedOrginizations[value.Name] = value
	}

	return mappedOrginizations
}

// maps all zones to their names
// NOTE: takes an assumption that unique names are enforced within the database
func mapAllZones(zones []polygon_initial.Zone) map[string]polygon_initial.Zone {
	mappedZones := make(map[string]polygon_initial.Zone)
	for _, value := range zones {
		if _, ok := mappedZones[value.Name]; ok { // if the key already exists
			panic("Duplicate zone Name")
		}
		mappedZones[value.Name] = value
	}

	return mappedZones
}

// normalizes the differences between values stored in the database and in the csv values
func normalizeOrganizationNames(organizations []polygon_initial.ApiOrganization) []polygon_initial.ApiOrganization {
	for _, value := range organizations {
		switch value.Name {
		case "Feb Distributing":
			value.Name = "F.E.B. Distributing"
		case "Trieagle":
			value.Name = "Tri Eagle"
		case "Verst Logistics":
			value.Name = "Verst"
		case "Wn Foods":
			value.Name = "WN Foods"
		}
	}

	return organizations
}

func isSandbox(record []string) bool {
	return strings.ToLower(record[2]) != "customer"
}

func fetchLifecycle(deviceType string, record []string) string {
	if deviceType == "edge" {
		switch record[5] {
		case "Offline ":
			return "maintenance"
		case "Online":
			return "live"
		default:
			return DEFAULT_EDGE_LIFECYCLE
		}
	} else if deviceType == "camera" {
		switch record[6] {
		case "Troubleshooting":
			return "maintenance"
		case "Online":
			return "live"
		default:
			return DEFAULT_CAMERA_LIFECYCLE
		}
	}
	return ""
}

func run() error {
	ctx := context.Background()

	db, err := sql.Open("postgres", fmt.Sprintf("host=%v port=%v dbname=%v user=%v password=%v sslmode=disable",
		DATABASE_HOST,
		DATABASE_PORT,
		DATABASE_NAME,
		DATABASE_USERNAME,
		DATABASE_PASSWORD))
	if err != nil {
		log.Panicf("error opening database: %v", err)
	}

	queries := polygon_initial.New(db)

	// fetch and map all organizations
	allOrganizations, err := queries.FetchAllOrganizations(ctx)
	if err != nil {
		return fmt.Errorf("error fetching all organizations: %v", err)
	}
	normalizeOrganizationNames(allOrganizations)
	mappedOrgs := mapAllOrganizations(allOrganizations)

	// fetch all the zones
	allZones, err := queries.FetchTopLevelZones(ctx)
	if err != nil {
		return fmt.Errorf("error fetching all zones: %v", err)
	}
	mappedZones := mapAllZones(allZones)

	// open csv file
	file, err := os.Open(CSV_FILE)
	if err != nil {
		return fmt.Errorf("error opening file: %v", err)
	}

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("error reading csv file: %v", err)
	}

	// skip the header
	records = records[1:]

	// iterate through CSV
	for recordNumber, record := range records {
		processedRecord := processedRecord{
			Record: record,
		}
		// begin a transaction per record. If the record fails, the transaction will be rolled back
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			log.Panicf("Failed to begin transaction: %v", err)
		}

		// termination function which rollsback the transaction if an error occurs
		termimnate := func(err error) error {
			rollbackErr := tx.Rollback()
			if rollbackErr != nil {
				err = fmt.Errorf("failed to rollback transaction in response to %v.\n\nerror rolling back transaction: %w", err, rollbackErr)
			}
			return err
		}
		// collect all values from the csv row
		orgName := record[0]
		orgId := int64(-1)
		org, orgExists := mappedOrgs[orgName]
		if orgExists {
			orgId = org.ID
		}
		zoneName := record[1]
		zoneId := int64(-1)
		zone, zoneExists := mappedZones[zoneName]
		if zoneExists {
			zoneId = zone.ID
		}
		edgeLifecycle := fetchLifecycle("edge", record)
		cameraLifecycle := fetchLifecycle("camera", record)
		// csv column 'SN & Landscape'
		sn := record[7]
		// csv column 'Edge Appliance UUID & Configuration'
		// logger := log.Logger()
		//logger := log.Default()

		log.Printf("record: %v", record[8])
		uid, err := uuid.Parse(record[8])
		if err != nil {
			return termimnate(fmt.Errorf("Failed to parse UUID: %w", err))
		}

		log.Printf(`Using the following variables derived from the CSV row:
			orgName: %s
			orgId: %d 
			zoneName: %s
			zoneId: %d 
			edgeLifecycle: %s 
			cameraLifecycle: %s 
			sn: %s 
			uid: %s
			`, orgName, orgId, zoneName, zoneId, edgeLifecycle, cameraLifecycle, sn, uid)

		// validate that the orginizaiton exists
		if !orgExists {
			org, err = queries.CreateOrganization(ctx, polygon_initial.CreateOrganizationParams{
				Name:      orgName,
				Key:       strings.ToUpper(strings.Replace(orgName, " ", "_", -1)),
				IsSandbox: isSandbox(record),
				Timezone:  "invalid", // will update once the values are provided in the csv document
			})
			if err != nil {
				return termimnate(fmt.Errorf("Failed to create organization: %w", err))
			}

			mappedOrgs[org.Name] = org
			orgId = org.ID
		}
		processedRecord.Organization = org

		// validate that zone exists
		if !zoneExists {
			zone, err = queries.CreateZone(ctx, polygon_initial.CreateZoneParams{
				Name:           zoneName,
				Key:            strings.ToUpper(strings.Replace(zoneName, " ", "_", -1)),
				ZoneType:       "site", // will update once the values are provided in the csv document
				OrganizationID: mappedOrgs[orgName].ID,
			})
			if err != nil {
				return termimnate(fmt.Errorf("Failed to create zone: %w", err))
			}
			mappedZones[zone.Name] = zone
			zoneId = zone.ID
		}
		processedRecord.Zone = zone

		// create the edge device
		edge, err := queries.CreateEdge(ctx, polygon_initial.CreateEdgeParams{
			Uuid:           uid,
			OrganizationID: sql.NullInt64{Int64: orgId, Valid: true}, // bug in portal code
			Name:           autoname.Generate("-"),
			Serial:         sql.NullString{String: sn, Valid: true},
			Lifecycle:      edgeLifecycle,
		})
		if err != nil {

			if pqErr, ok := err.(*pq.Error); ok {
				log.Printf("Failed to create edge device: %v", err)
				// pqErr := err.(*pq.Error)
				if pqErr.Code == "23505" {
					log.Printf("Edge with UUID %s already exists. Skipping.", uid)
					continue
				}
			}

			return termimnate(fmt.Errorf("Failed to create edge: %w", err))
		}
		processedRecord.Edge = edge

		// create the camera
		cameras, err := queries.UpdateCameras(ctx, polygon_initial.UpdateCamerasParams{
			Lifecycle:      sql.NullString{String: cameraLifecycle, Valid: true},
			EdgeID:         sql.NullInt64{Int64: edge.ID, Valid: true},
			ZoneID:         sql.NullInt64{Int64: zoneId, Valid: true},
			OrganizationID: sql.NullInt64{Int64: orgId, Valid: true},
		})
		processedRecord.Cameras = cameras

		if err != nil {
			return termimnate(fmt.Errorf("Failed to update cameras: %w", err))
		}

		// write to yml
		fileName := fmt.Sprintf(YAML_FILE_NAME, recordNumber)
		log.Printf("Writing to yml file: %s", fileName)
		processedDataYaml, err := processedRecord.toYaml()
		if err != nil {
			return termimnate(fmt.Errorf("error marshalling yaml: %w", err))
		}
		err = ioutil.WriteFile(fileName, processedDataYaml, 0644)
		if err != nil {
			return termimnate(fmt.Errorf("error writing yaml to %v: %w", fileName, err))
		}

		// close the transaction
		err = tx.Commit()
		if err != nil {
			return fmt.Errorf("Failed to commit transaction: %w", err)
		}
	}

	return nil
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}
