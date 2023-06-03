import React, { useMemo, useState, useEffect } from "react";
import {
  GetFilterBarData,
  GetFilterBarData_currentUser_site_assignableUsers_edges_node,
  GetFilterBarData_currentUser_site_cameras_edges_node,
} from "__generated__/GetFilterBarData";
import { useQuery } from "@apollo/client";
import { GET_FILTER_BAR_DATA } from "./queries";
import { getNodes } from "graphql/utils";
import {
  FilterBag,
  FilterSelectionOption,
  FilterFormSectionOptionGroup,
  FilterFormSection,
  FILTER_STATUSES,
  FILTER_PRIORITIES,
  FILTER_KEY_INCIDENT_TYPE,
  FILTER_KEY_ASSIGNMENT,
  FILTER_KEY_CAMERA,
  FILTER_VALUE_ASSIGNED_TO_ME,
  FILTER_VALUE_ASSIGNED_BY_ME,
  FILTER_EXTRAS,
  FilterConfig,
} from ".";

interface UserNode extends GetFilterBarData_currentUser_site_assignableUsers_edges_node {}
interface CameraNode extends GetFilterBarData_currentUser_site_cameras_edges_node {}
interface FilterFormProps {
  values: FilterBag;
  config: FilterConfig;
  onChange: (values: FilterBag) => void;
}

export function FilterForm(props: FilterFormProps) {
  const [currentValues, setCurrentValues] = useState<FilterBag>({ ...props.values });
  const { config } = props;
  const { data } = useQuery<GetFilterBarData>(GET_FILTER_BAR_DATA);
  const incidentTypeSection = useMemo(() => {
    const filtered = (data?.currentUser?.organization?.incidentTypes || []).filter((incidentType) => !!incidentType);
    const options = filtered.map((incidentType) => ({
      label: incidentType?.name,
      value: incidentType?.key,
    })) as FilterSelectionOption[];
    return {
      title: "Incident Types",
      key: FILTER_KEY_INCIDENT_TYPE,
      options: options,
    };
  }, [data]);

  const cameraSection = useMemo(() => {
    const options = getNodes<CameraNode>(data?.currentUser?.site?.cameras).map((camera) => ({
      label: camera?.name,
      value: camera?.id,
    })) as FilterSelectionOption[];
    return {
      title: "Cameras",
      key: FILTER_KEY_CAMERA,
      options: options,
    };
  }, [data]);

  const assignmentSection = useMemo(() => {
    const presetGroup: FilterFormSectionOptionGroup = {
      key: "preset-option-group",
      options: [
        // TODO(itay): If currentUser.is_assignable == False => Remove "Assigned to Me" option.
        {
          label: "Assigned To Me",
          value: FILTER_VALUE_ASSIGNED_TO_ME,
        },
        {
          label: "Assigned By Me",
          value: FILTER_VALUE_ASSIGNED_BY_ME,
        },
      ],
    };
    const usersGroup: FilterFormSectionOptionGroup = {
      key: "user-option-group",
      options: getNodes<UserNode>(data?.currentUser?.site?.assignableUsers).map((user) => ({
        label: user.fullName,
        value: user.id,
      })) as FilterSelectionOption[],
    };
    return {
      title: "Assignees",
      key: FILTER_KEY_ASSIGNMENT,
      groups: [presetGroup, usersGroup],
    };
  }, [data]);

  const sections = useMemo(
    () => [
      {
        section: FILTER_STATUSES,
        visible: config.allFilters,
      },
      { section: FILTER_PRIORITIES, visible: config.allFilters },
      { section: cameraSection, visible: config.locations },
      { section: assignmentSection, visible: config.allFilters },
      { section: incidentTypeSection, visible: config.allFilters },
      { section: FILTER_EXTRAS, visible: config.allFilters },
    ],
    [config, cameraSection, assignmentSection, incidentTypeSection]
  );

  useEffect(() => {
    setCurrentValues(props.values);
  }, [props.values]);

  return (
    <div>
      {sections.map((target, idx) => {
        if (target.visible) {
          return (
            <div className={"border-gray-300 pb-4 mb-4 border-b last:mb-0 last:border-b-0"} key={target.section.key}>
              <FilterFormSection
                onChange={(values) => props.onChange({ ...values })}
                values={currentValues}
                section={target.section}
              />
            </div>
          );
        }
        return <></>;
      })}
    </div>
  );
}
