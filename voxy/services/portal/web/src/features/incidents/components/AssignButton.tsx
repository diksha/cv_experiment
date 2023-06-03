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
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import classNames from "classnames";
import { getNodes } from "graphql/utils";
import { Modal, ModalBody, MultiSelect } from "ui";
import { Button, SelectChangeEvent } from "@mui/material";
import { X, UserPlus } from "phosphor-react";
import { useMutation, useQuery } from "@apollo/client";
import { GET_INCIDENT_COMMENTS, GET_INCIDENT_DETAILS, ASSIGN_INCIDENT, UNASSIGN_INCIDENT } from "features/incidents";
import { GET_CURRENT_SITE_ASSIGNABLE_USERS } from "features/organizations";
import { GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges_node } from "__generated__/GetCurrentSiteAssignableUsers";
import { GetIncidentDetails, GetIncidentDetailsVariables } from "__generated__/GetIncidentDetails";
import { GetComments } from "__generated__/GetComments";
import { GetCurrentSiteAssignableUsers } from "__generated__/GetCurrentSiteAssignableUsers";
import { GetCommentsVariables } from "__generated__/GetComments";
import { AssignIncident, AssignIncidentVariables } from "__generated__/AssignIncident";
import { UnassignIncident, UnassignIncidentVariables } from "__generated__/UnassignIncident";

interface UserNode extends GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges_node {}

interface AssigneeName {
  id?: string;
  fullName: string;
  initials?: string | null;
}

interface AssigneeProps extends AssigneeName {
  hideFullNameOnMobile?: boolean;
}

export function Assignee(props: AssigneeProps) {
  return (
    <div className="flex gap-1">
      <div className="grid text-xs text-center content-center h-7 w-7 font-bold bg-gray-200 rounded-full">
        {props.initials}
      </div>
      <div className={classNames("content-center", props.hideFullNameOnMobile ? "hidden md:grid" : "grid")}>
        {props.fullName}
      </div>
    </div>
  );
}

export function AssigneeList(props: { assignees?: AssigneeName[]; hideFullNameOnMobile?: boolean }) {
  const assigneesToShow = 2;
  const allAssignees = props.assignees || [];
  const visibleAssignees = allAssignees.slice(0, assigneesToShow);
  const hiddenAssignees = allAssignees.slice(assigneesToShow);
  if (hiddenAssignees.length > 0) {
    visibleAssignees.push({
      id: "more-participants",
      initials: `+${hiddenAssignees.length}`,
      fullName: `more participant${hiddenAssignees.length > 1 ? "s" : ""}`,
    });
  }
  return visibleAssignees.length > 0 ? (
    <div className="flex gap-2 text-sm">
      {visibleAssignees.map((a) => (
        <Assignee
          key={a.id}
          fullName={a.fullName}
          initials={a.initials}
          hideFullNameOnMobile={props.hideFullNameOnMobile}
        />
      ))}
    </div>
  ) : (
    <div className="text-sm text-brand-gray-200">Unassigned</div>
  );
}
interface AssignButtonProps {
  incidentId: string;
  incidentUuid: string;
  incidentTitle?: string;
  assignedUserIds: string[];
}

export function AssignButton({ incidentId, incidentUuid, incidentTitle, assignedUserIds }: AssignButtonProps) {
  const [openModal, setOpenModal] = useState(false);
  const [note, setNote] = useState<string>("");
  const [selectedUserIds, setSelectedUserIds] = useState<string[]>([]);
  const [openConfirm, setOpenConfirm] = useState(false);
  const [unassignUser, setUnassignUser] = useState<UserNode>();
  let initFocus = useRef(null);

  const { refetch: refetchIncidentDetails } = useQuery<GetIncidentDetails, GetIncidentDetailsVariables>(
    GET_INCIDENT_DETAILS,
    {
      variables: {
        incidentUuid,
      },
    }
  );
  const { refetch: refetchComments } = useQuery<GetComments, GetCommentsVariables>(GET_INCIDENT_COMMENTS, {
    variables: {
      incidentId,
    },
  });
  const { data: siteUserData } = useQuery<GetCurrentSiteAssignableUsers>(GET_CURRENT_SITE_ASSIGNABLE_USERS);

  const allUsers = useMemo(() => {
    return getNodes<UserNode>(siteUserData?.currentUser?.site?.assignableUsers).filter(
      (user: UserNode) => !!user.fullName?.trim()
    );
  }, [siteUserData]);

  const assignedUsers = useMemo(() => {
    return allUsers.filter((user: UserNode) => assignedUserIds.includes(user.id));
  }, [assignedUserIds, allUsers]);

  const unassignedUsers = useMemo(() => {
    return allUsers.filter((user: UserNode) => !assignedUserIds.includes(user.id));
  }, [assignedUserIds, allUsers]);

  const refetch = useCallback(() => {
    refetchComments();
    refetchIncidentDetails();
  }, [refetchComments, refetchIncidentDetails]);

  const [mutateAssignIncident, { data: assignIncident }] = useMutation<AssignIncident, AssignIncidentVariables>(
    ASSIGN_INCIDENT
  );
  const [mutateUnassignIncident, { data: unassignIncident }] = useMutation<UnassignIncident, UnassignIncidentVariables>(
    UNASSIGN_INCIDENT
  );

  useEffect(() => {
    if (!!assignIncident || !!unassignIncident) {
      refetch();
    }
  }, [assignIncident, refetch, unassignIncident]);

  const isAssigned = assignedUserIds.length > 0;

  useEffect(() => {
    setSelectedUserIds([]);
  }, [openModal]);

  const handleAssign = () => {
    mutateAssignIncident({ variables: { incidentId, assigneeIds: selectedUserIds, note } });
    setOpenModal(false);
  };

  const handleOpenConfirmUnassign = (user: UserNode) => {
    setOpenConfirm(true);
    setUnassignUser(user);
  };

  const handleCloseConfirmUnassign = () => {
    setOpenConfirm(false);
    setUnassignUser(undefined);
  };

  const handleUnassign = () => {
    if (unassignUser) {
      mutateUnassignIncident({
        variables: { incidentId, assigneeId: unassignUser.id },
      });
    }
    handleCloseConfirmUnassign();
  };

  const handleSelectChange = (e: SelectChangeEvent<(string | number)[]>) => {
    const value = e.target.value;
    const valueArray = Array.isArray(value) ? value : [value];
    const selected = valueArray.filter((id): id is string => typeof id === "string");
    setSelectedUserIds(selected);
  };

  return (
    <div>
      <div
        data-ui-key="button-assignees"
        className="flex justify-between cursor-pointer hover:text-yellow-600"
        onClick={() => setOpenModal(true)}
      >
        <h4 className="font-bold">Assignees</h4>
        <button>
          <UserPlus size={18} />
        </button>
      </div>
      <div className="flex flex-col gap-2">
        {!isAssigned ? (
          <div>None yet</div>
        ) : (
          assignedUsers.map((user: UserNode) => (
            <button
              className={classNames("flex w-full text-left items-center", "hover:text-yellow-600 cursor-pointer")}
              key={user.id}
              onClick={() => handleOpenConfirmUnassign(user)}
            >
              <div className="flex-grow">
                <Assignee fullName={user.fullName || ""} initials={user.initials} />
              </div>
              <div>
                <X size={18} />
              </div>
            </button>
          ))
        )}
      </div>
      <Modal open={openConfirm} onClose={handleCloseConfirmUnassign}>
        <ModalBody>
          <div>
            <div className="mb-6">
              Are you sure you want to unassign <strong>{unassignUser?.fullName}</strong>?
            </div>
            <div className="flex gap-4">
              <Button className="flex-grow-0" variant="outlined" onClick={handleCloseConfirmUnassign}>
                Cancel
              </Button>
              <Button className="flex-grow" variant="contained" onClick={handleUnassign}>
                Unassign
              </Button>
            </div>
          </div>
        </ModalBody>
      </Modal>
      <Modal open={openModal} onClose={() => setOpenModal(false)}>
        <ModalBody>
          <div className="flex flex-col">
            {incidentTitle ? <div className="text-2xl">Assign incident</div> : null}
            <div style={{ margin: "10px 0px" }}>
              <MultiSelect
                FormControlProps={{
                  disabled: allUsers.length === 0,
                }}
                SelectProps={{
                  onChange: handleSelectChange,
                  renderValue: (selected: (string | number)[]) =>
                    selected.length ? `Selected (${selected.length})` : "Assign to...",
                  value: selectedUserIds,
                }}
                items={unassignedUsers}
                valueKey="id"
                labelKey="fullName"
              />
            </div>
            <div style={{ margin: "10px 0px" }}>
              <label className="block text-sm font-bold text-gray-700">
                Note&nbsp;
                <span className="text-gray-400">(visible to all team members)</span>
              </label>
              <textarea
                data-ui-key="textarea-assign-to-note"
                value={note}
                className="w-full rounded-md border-gray-300"
                ref={initFocus}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Provide additional details here..."
                disabled={allUsers.length === 0}
              />
            </div>
          </div>

          <div className="flex gap-4">
            <Button
              data-ui-key="button-assign-to-cancel"
              className="flex-grow-0"
              variant="outlined"
              onClick={() => setOpenModal(false)}
            >
              Cancel
            </Button>
            <Button
              data-ui-key="button-assign-to-submit"
              className="flex-grow"
              disabled={selectedUserIds.length === 0 || allUsers.length === 0}
              variant="contained"
              onClick={handleAssign}
            >
              Assign
            </Button>
          </div>
        </ModalBody>
      </Modal>
    </div>
  );
}
