/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */

import { useTheme } from "@mui/material";
// trunk-ignore-all(gitleaks/generic-api-key): no API keys here, just SVG paths

type LogoPart = "top" | "bottomLeft" | "bottomRight";
type LogoVariant = "dark" | "light" | "purple";
type LogoColorMap = Record<LogoVariant, Record<LogoPart, string>>;

export function Logo(props: { style?: "light" | "dark" }) {
  const variant = props.style || "dark";
  const logoEl = variant === "dark" ? <DarkLogo /> : <LightLogo />;

  return <span>{logoEl}</span>;
}

interface LogoIconProps {
  variant?: LogoVariant;
}
export function LogoIcon({ variant = "dark" }: LogoIconProps) {
  const theme = useTheme();
  const logoColors: LogoColorMap = {
    dark: {
      top: theme.palette.primary[900],
      bottomLeft: theme.palette.primary[900],
      bottomRight: theme.palette.primary[900],
    },
    light: {
      top: "#ffffff",
      bottomLeft: "#ffffff",
      bottomRight: "#ffffff",
    },
    purple: {
      top: "#C1AAE7",
      bottomLeft: "#632BC2",
      bottomRight: "#632BC2",
    },
  };
  return (
    <svg className="h-full w-full" viewBox="0 0 34 38" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M7.54859 12.3592L16.9996 6.90503L26.4505 12.3591L32.433 8.90665L16.9996 0L1.56616 8.90673L7.54859 12.3592Z"
        fill={logoColors[variant]["top"]}
      />
      <path
        d="M16.1358 30.1775L6.70247 24.7335V13.8658L0.719971 10.4132V28.1859L16.1358 37.0822V30.1775Z"
        fill={logoColors[variant]["bottomLeft"]}
      />
      <path
        d="M27.2975 13.8658V24.7335L17.8642 30.1775V37.0822L33.28 28.1859V10.4132L27.2975 13.8658Z"
        fill={logoColors[variant]["bottomRight"]}
      />
    </svg>
  );
}

function DarkLogo() {
  return (
    <svg className="h-full w-full" viewBox="0 0 500 160" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g clipPath="url(#clip0_209_754)">
        <path
          d="M28.4551 54.7673L67.8378 32.0398L107.22 54.767L132.149 40.3806L67.8378 3.26636L3.52618 40.3809L28.4551 54.7673Z"
          fill="#CFBDEB"
        />
        <path d="M64.2382 129.017L24.9292 106.332V61.0455L0 46.6585V120.718L64.2382 157.789V129.017Z" fill="#632BC2" />
        <path
          d="M110.749 61.0455V106.332L71.4404 129.017V157.789L135.678 120.718V46.6585L110.749 61.0455Z"
          fill="#632BC2"
        />
        <path
          d="M200.885 97.1147L216.165 45.1419H233.755L210.371 115.812H192.561L169.598 45.1419H187.406L202.359 97.1147H200.885Z"
          fill="#632BC2"
        />
        <path
          d="M270.953 117.496C264.238 117.496 258.381 116.084 253.379 113.259C248.375 110.43 244.487 106.252 241.712 100.727C238.942 95.1944 237.555 88.3755 237.552 80.2707C237.552 71.9412 238.939 65.0687 241.714 59.6533C244.493 54.2294 248.382 50.1898 253.381 47.534C258.384 44.8892 264.242 43.5642 270.954 43.559C277.727 43.559 283.635 44.9458 288.678 47.7194C293.716 50.4986 297.621 54.6058 300.395 60.041C303.167 65.4901 304.554 72.2334 304.557 80.2709C304.557 88.5161 303.169 95.4054 300.395 100.939C297.624 106.464 293.718 110.604 288.678 113.361C283.631 116.123 277.723 117.502 270.953 117.496ZM270.953 102.387C275.864 102.387 279.795 100.447 282.746 96.567C285.694 92.6844 287.168 87.2157 287.168 80.1607C287.168 73.5241 285.694 68.2826 282.746 64.4366C279.799 60.5962 275.867 58.6759 270.953 58.6759C266.177 58.6759 262.312 60.5962 259.362 64.4366C256.414 68.2855 254.94 73.5634 254.94 80.2701C254.94 87.2519 256.414 92.6842 259.362 96.5668C262.31 100.441 266.173 102.381 270.953 102.387V102.387Z"
          fill="#632BC2"
        />
        <path
          d="M390.618 115.812V45.142H437.868V59.7289H407.474V73.3138H435.972V87.9514H407.474V101.064H437.868V115.812H390.618Z"
          fill="#632BC2"
        />
        <path d="M472.041 45.142V101.064H498.744V115.812H455.184V45.142H472.041Z" fill="#632BC2" />
        <path d="M308.11 115.814H328.546L377.059 45.142H356.726L308.11 115.814Z" fill="#632BC2" />
        <path d="M340.383 62.4984L328.443 45.142H308.11L330.194 77.3115L340.383 62.4984Z" fill="#632BC2" />
        <path d="M354.966 83.7232L344.763 98.5601L356.608 115.816H377.044L354.966 83.7232Z" fill="#632BC2" />
      </g>
      <defs>
        <clipPath id="clip0_209_754">
          <rect width="500" height="153.266" fill="white" transform="translate(0 3.26636)" />
        </clipPath>
      </defs>
    </svg>
  );
}

function LightLogo() {
  return (
    <svg className="h-full w-full" viewBox="0 0 500 160" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g clipPath="url(#clip0_215_764)">
        <path
          d="M28.4551 54.7673L67.8378 32.0398L107.22 54.767L132.149 40.3806L67.8378 3.26636L3.52618 40.3809L28.4551 54.7673Z"
          fill="#E5E7EB"
        />
        <path d="M64.2382 129.017L24.9292 106.332V61.0455L0 46.6585V120.718L64.2382 157.789V129.017Z" fill="white" />
        <path
          d="M110.749 61.0455V106.332L71.4404 129.017V157.789L135.678 120.718V46.6585L110.749 61.0455Z"
          fill="white"
        />
        <path
          d="M200.885 97.1147L216.165 45.1419H233.755L210.371 115.812H192.561L169.598 45.1419H187.406L202.359 97.1147H200.885Z"
          fill="white"
        />
        <path
          d="M270.953 117.496C264.238 117.496 258.381 116.084 253.379 113.259C248.375 110.43 244.487 106.252 241.712 100.727C238.942 95.1944 237.555 88.3755 237.552 80.2707C237.552 71.9412 238.939 65.0687 241.714 59.6533C244.493 54.2294 248.382 50.1898 253.381 47.534C258.384 44.8892 264.242 43.5642 270.954 43.559C277.727 43.559 283.635 44.9458 288.678 47.7194C293.716 50.4986 297.621 54.6058 300.395 60.041C303.167 65.4901 304.554 72.2334 304.557 80.2709C304.557 88.5161 303.169 95.4054 300.395 100.939C297.624 106.464 293.718 110.604 288.678 113.361C283.631 116.123 277.723 117.502 270.953 117.496ZM270.953 102.387C275.864 102.387 279.795 100.447 282.746 96.567C285.694 92.6844 287.168 87.2157 287.168 80.1607C287.168 73.5241 285.694 68.2826 282.746 64.4366C279.799 60.5962 275.867 58.6759 270.953 58.6759C266.177 58.6759 262.312 60.5962 259.362 64.4366C256.414 68.2855 254.94 73.5634 254.94 80.2701C254.94 87.2519 256.414 92.6842 259.362 96.5668C262.31 100.441 266.173 102.381 270.953 102.387V102.387Z"
          fill="white"
        />
        <path
          d="M390.618 115.812V45.142H437.868V59.7289H407.474V73.3138H435.972V87.9514H407.474V101.064H437.868V115.812H390.618Z"
          fill="white"
        />
        <path d="M472.041 45.142V101.064H498.744V115.812H455.184V45.142H472.041Z" fill="white" />
        <path d="M308.11 115.814H328.546L377.059 45.142H356.726L308.11 115.814Z" fill="white" />
        <path d="M340.383 62.4984L328.443 45.142H308.11L330.194 77.3115L340.383 62.4984Z" fill="white" />
        <path d="M354.966 83.7232L344.763 98.5601L356.608 115.816H377.044L354.966 83.7232Z" fill="white" />
      </g>
      <defs>
        <clipPath id="clip0_215_764">
          <rect width="500" height="153.266" fill="white" transform="translate(0 3.26636)" />
        </clipPath>
      </defs>
    </svg>
  );
}
