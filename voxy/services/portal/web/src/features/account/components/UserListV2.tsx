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
import { useQuery, useMutation } from "@apollo/client";
import { DateTime } from "luxon";
import classNames from "classnames";
import { useCurrentUser, CurrentUser } from "features/auth";
import { getNodes } from "graphql/utils";
import { EditRoleModal, EditSiteModal, RemoveUserModal, USER_RESEND_INVITATION } from "features/account";
import React, { useMemo, useState, Fragment, useEffect, useCallback } from "react";
import { Avatar, Spinner } from "ui";
import { Pagination } from "@mui/material";
import { GET_CURRENT_USER_TEAMMATES } from "features/organizations";
import { USERS_REMOVE, USERS_UPDATE_ROLE, USERS_UPDATE_SITE } from "features/auth";
import {
  GetCurrentUserTeammates,
  GetCurrentUserTeammates_currentUser_teammates_edges_node,
} from "__generated__/GetCurrentUserTeammates";
import { Menu, Transition } from "@headlessui/react";
import { DotsThreeCircleVertical, Lock, Info } from "phosphor-react";
import { UserResendInvitation, UserResendInvitationVariables } from "__generated__/UserResendInvitation";

const ITEMS_PER_PAGE = 10;

interface UserNode extends GetCurrentUserTeammates_currentUser_teammates_edges_node {
  invitation?: InvitationNode;
}

interface InvitationNode {
  createdAt: DateTime;
  expired: boolean;
  token: string;
}

export function OrganizationUserListV2() {
  const { currentUser } = useCurrentUser();
  const { data, loading } = useQuery<GetCurrentUserTeammates>(GET_CURRENT_USER_TEAMMATES);

  const users: UserNode[] = useMemo(() => {
    return getNodes<UserNode>(data?.currentUser?.teammates);
  }, [data]);

  const invitedUsers: UserNode[] = [];
  if (data?.currentUser?.invitedUsers) {
    for (let invitedUser of data?.currentUser?.invitedUsers) {
      invitedUser.user &&
        invitedUsers.push({
          id: invitedUser.user?.id,
          email: invitedUser.user?.email,
          sites: invitedUser.sites,
          roles: [invitedUser.role],
          __typename: "UserType",
          firstName: "",
          lastName: "",
          fullName: null,
          isActive: false,
          invitation: {
            createdAt: DateTime.fromISO(invitedUser.createdAt),
            expired: invitedUser.expired,
            token: invitedUser.token,
          },
        });
    }
  }

  return <UserListV2 loading={loading} users={[...invitedUsers, ...users]} currentUser={currentUser} />;
}

export function UserListV2(props: { loading: boolean; users: UserNode[]; currentUser?: CurrentUser }) {
  const [currentPage, setCurrentPage] = useState(1);
  const showData = !props.loading && props.currentUser && props.users.length > 0;
  const showSpinner = props.loading;
  const spinnerClasses = classNames("grid justify-center opacity-40", showData ? "p-8" : "p-36");
  // TODO: need to move this to backend once we have a lot of users per org
  const pageCount = Math.ceil(props.users.length / ITEMS_PER_PAGE);
  const currentPageUsers = props.users.slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE);

  const handlePageChange = (_: React.ChangeEvent<unknown>, value: number) => {
    setCurrentPage(value);
  };

  return (
    <div className="flex flex-col">
      <table className="table-auto min-w-full">
        <thead className="bg-white hidden lg:table-header-group text-left text-xs font-medium text-brand-gray-300 tracking-wider">
          <tr>
            <th scope="col" className="w-2/5 py-3">
              Name
            </th>
            <th scope="col" className="w-1/5 py-3">
              Role
            </th>
            <th scope="col" className="w-1/5 py-3">
              Site
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-brand-gray-050">
          {showData ? (
            <>
              {currentPageUsers.map((user: UserNode, idx: number) =>
                user ? <Row key={idx} user={user} /> : <div>Invalid user</div>
              )}
            </>
          ) : null}
        </tbody>
      </table>
      {showSpinner ? (
        <div className={spinnerClasses}>
          <div>
            <Spinner />
          </div>
        </div>
      ) : (
        <div className="flex justify-center py-6">
          <Pagination page={currentPage} count={pageCount} onChange={handlePageChange} />
        </div>
      )}
    </div>
  );
}

function Row(props: { user: UserNode; onClick?: () => void }) {
  const { currentUser } = useCurrentUser();

  const [isSiteModalOpen, setIsSiteModalOpened] = useState<boolean>(false);
  const [isRoleModalOpen, setIsRoleModalOpened] = useState<boolean>(false);
  const [isRemoveModalOpened, setIsRemoveModalOpened] = useState<boolean>(false);

  const [reinviteUser, { loading }] = useMutation<UserResendInvitation, UserResendInvitationVariables>(
    USER_RESEND_INVITATION,
    {
      refetchQueries: [GET_CURRENT_USER_TEAMMATES],
    }
  );

  const [visibleSites, setVisibleSites] = useState("");

  const handleReinviteUser = useCallback(() => {
    if (props.user.invitation) {
      reinviteUser({
        variables: {
          invitationToken: props.user.invitation.token,
        },
      });
    }
  }, [props.user.invitation, reinviteUser]);

  useEffect(() => {
    if (props.user.sites?.length) {
      setVisibleSites(
        props.user.sites
          .slice(0, 3)
          .map((site) => site!.name)
          .join(", ")
      );
    }
  }, [props.user.sites, setVisibleSites]);

  const showAllSites = useCallback(() => {
    if (props.user.sites?.length) {
      setVisibleSites(props.user.sites.map((site) => site!.name).join(", "));
    }
  }, [props.user.sites]);

  const handleModalClose = useCallback((modalStateToggle: (value: boolean) => void): (() => void) => {
    return () => {
      modalStateToggle(false);
    };
  }, []);

  const roleList = useMemo(() => {
    if (!props.user.roles?.length) {
      return "";
    }
    if (props.user.roles?.length === 1) {
      return props.user.roles[0].name;
    }
    return props.user.roles?.map((role) => role.name).join(", ");
  }, [props.user]);

  const resendText = (viewport: string) =>
    props.user.invitation?.expired ? (
      <div
        className={`relative text-left ${viewport === "mobile" ? "m-2 md:hidden text-sm" : "hidden md:block text-xs"}`}
      >
        {loading ? (
          <Spinner />
        ) : (
          <div className="flex items-center">
            <Info size={24} className="mx-1" />
            <div>
              Invited {props.user.invitation.createdAt.toRelative()},{" "}
              <span className="cursor-pointer font-bold text-brand-purple-500" onClick={handleReinviteUser}>
                Resend Invite ?
              </span>
            </div>
          </div>
        )}
      </div>
    ) : null;

  return (
    <>
      <tr key={props.user.id}>
        <td className="py-4 pl-4 md:pl-0 whitespace-nowrap">
          <div className="flex items-center gap-4">
            <div className="flex-shrink-0 h-10 w-10">
              <Avatar url={undefined} name={props.user.fullName!} />
            </div>
            <div>
              {props.user.fullName ? (
                <div className="font-semibold text-brand-gray-500 font-epilogue flex">{props.user.fullName}</div>
              ) : null}
              <div className="flex gap-2">
                <div className="flex items-center text-sm text-brand-gray-300">{props.user.email}</div>
                {/* if inactive it means that they are invited users */}
                {props.user.isActive ? null : <InvitedBadge />}
              </div>
            </div>
            {resendText("mobile")}
          </div>
        </td>
        <td className="py-4 pr-8 whitespace-nowrap hidden lg:table-cell">
          <div className="whitespace-normal">{roleList}</div>
        </td>
        <td className="py-4 whitespace-nowrap table-cell">
          <div className="flex items-center justify-between lg:gap-4 whitespace-normal">
            {props.user.sites?.length ? (
              <div className="hidden lg:block text-sm overflow-hidden text-ellipsis text-brand-gray-500">
                {visibleSites}
                {props.user.sites.length !== visibleSites.split(", ").length && (
                  <div className="hover:cursor-pointer text-brand-gray-500 font-bold" onClick={showAllSites}>
                    View More
                  </div>
                )}
              </div>
            ) : (
              <div className="hidden lg:block" />
            )}
            {props.user.isActive ? (
              <div className="relative text-left">
                <UserEditMenu
                  setIsRoleModalOpened={setIsRoleModalOpened}
                  setIsSiteModalOpened={setIsSiteModalOpened}
                  setIsRemoveModalOpened={setIsRemoveModalOpened}
                  user={props.user}
                />
              </div>
            ) : (
              resendText("desktop")
            )}
          </div>
          <EditSiteModal
            isSiteModalOpen={isSiteModalOpen}
            selectedUser={props.user}
            onClose={handleModalClose(setIsSiteModalOpened)}
          />
          <EditRoleModal
            selectedUser={props.user}
            fullName={props.user.fullName || "Unknown"}
            isRoleModalOpen={isRoleModalOpen}
            roles={currentUser?.organization?.roles || []}
            onClose={handleModalClose(setIsRoleModalOpened)}
          />
          <RemoveUserModal
            isRemoveModalOpened={isRemoveModalOpened}
            onClose={handleModalClose(setIsRemoveModalOpened)}
            selectedUser={props.user}
          />
        </td>
      </tr>
    </>
  );
}

interface UserEditMenuItemProps {
  enabled: boolean;
  onClick: () => void;
  label: string;
}

/**
 * React.forwardRef is required here because HeadlessUI expects menu item children to accept a ref.
 */
const UserEditMenuItem = React.forwardRef<HTMLButtonElement, UserEditMenuItemProps>((props, ref) => {
  return (
    <button
      ref={ref}
      className={classNames("w-full flex gap-4 items-center px-6 py-3 text-left", {
        "cursor-pointer hover:bg-brand-gray-050": props.enabled,
        "disabled cursor-not-allowed text-brand-gray-300": !props.enabled,
      })}
      onClick={props.onClick}
      disabled={!props.enabled}
    >
      <div className="flex-1 font-bold">{props.label}</div>
      {!props.enabled ? <Lock className="h-4 w-4" /> : null}
    </button>
  );
});

interface UserEditMenuProps {
  user: UserNode | null;
  setIsRoleModalOpened: (value: boolean) => void;
  setIsSiteModalOpened: (value: boolean) => void;
  setIsRemoveModalOpened: (value: boolean) => void;
}
const UserEditMenu = ({
  setIsRoleModalOpened,
  setIsSiteModalOpened,
  setIsRemoveModalOpened,
  user,
}: UserEditMenuProps) => {
  const { currentUser } = useCurrentUser();
  const allowEditRole = !!currentUser?.hasZonePermission(USERS_UPDATE_ROLE);
  const allowEditSite = !!currentUser?.hasZonePermission(USERS_UPDATE_SITE);
  const allowRemove = !!currentUser?.hasZonePermission(USERS_REMOVE);

  const handleModalOpen = useCallback((modalStateToggle: (value: boolean) => void): (() => void) => {
    return () => {
      modalStateToggle(true);
    };
  }, []);

  return (
    <Menu as="div">
      <Menu.Button>
        <DotsThreeCircleVertical className="h-6 w-6 cursor-pointer pointer-events-auto" />
      </Menu.Button>
      <Transition
        as={Fragment}
        enter="transition ease-out duration-100"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items
          className={classNames(
            "absolute right-0 bg-white border-brand-gray-050 whitespace-nowrap font-medium",
            "origin-top-right divide-y divide-brand-gray-050 rounded-lg shadow-lg",
            "overflow-hidden ring-1 ring-black ring-opacity-5 focus:outline-none z-10"
          )}
        >
          <Menu.Item>
            <UserEditMenuItem
              enabled={allowEditRole}
              onClick={handleModalOpen(setIsRoleModalOpened)}
              label="Edit Role"
            />
          </Menu.Item>
          <Menu.Item>
            <UserEditMenuItem
              enabled={allowEditSite}
              onClick={handleModalOpen(setIsSiteModalOpened)}
              label="Edit Site"
            />
          </Menu.Item>
          <Menu.Item>
            <UserEditMenuItem
              enabled={allowRemove}
              onClick={handleModalOpen(setIsRemoveModalOpened)}
              label="Remove from Voxel"
            />
          </Menu.Item>
        </Menu.Items>
      </Transition>
      <div />
    </Menu>
  );
};

function InvitedBadge() {
  return (
    <div
      className={classNames(
        "inline-flex p-1 text-center font-bold rounded-lg",
        "items-center justify-center align-middle bg-brand-orange-500 text-white"
      )}
    >
      <span className="text-xs whitespace-nowrap px-1 font-normal">Invited</span>
    </div>
  );
}
