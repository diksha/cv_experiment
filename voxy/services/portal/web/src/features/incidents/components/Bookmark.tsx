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
import React, { useState } from "react";
import { BookmarkSimple } from "phosphor-react";
import classNames from "classnames";
import { useMutation } from "@apollo/client";
import { ADD_BOOKMARK, REMOVE_BOOKMARK } from "features/incidents";
import { CurrentUserAddBookmark, CurrentUserAddBookmarkVariables } from "__generated__/CurrentUserAddBookmark";
import { CurrentUserRemoveBookmark, CurrentUserRemoveBookmarkVariables } from "__generated__/CurrentUserRemoveBookmark";

export function Bookmark(props: { incidentId: string; bookmarked?: boolean }) {
  const [addBookmark, { loading: addBookmarkLoading }] = useMutation<
    CurrentUserAddBookmark,
    CurrentUserAddBookmarkVariables
  >(ADD_BOOKMARK);
  const [removeBookmark, { loading: removeBookmarkLoading }] = useMutation<
    CurrentUserRemoveBookmark,
    CurrentUserRemoveBookmarkVariables
  >(REMOVE_BOOKMARK);
  const [tempBookmarked, setTempBookmarked] = useState(props.bookmarked);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.nativeEvent.stopImmediatePropagation();
    if (tempBookmarked) {
      removeBookmark({ variables: { incidentId: props.incidentId } });
    } else {
      addBookmark({ variables: { incidentId: props.incidentId } });
    }
    setTempBookmarked(!tempBookmarked);
  };

  const iconClasses = classNames(
    "h-6 w-6 md:h-8 md:w-8 transition-colors",
    tempBookmarked
      ? "text-brand-yellow-300 hover:text-brand-yellow-400"
      : "text-brand-gray-100 hover:text-brand-gray-200"
  );

  return (
    <div>
      <button
        data-ui-key={tempBookmarked ? "button-remove-from-bookmarked-incidents" : "button-bookmark-this-incident"}
        onClick={handleClick}
        disabled={addBookmarkLoading || removeBookmarkLoading}
        title={tempBookmarked ? "Remove from bookmarked incidents" : "Bookmark this incident"}
      >
        <BookmarkSimple weight="fill" className={iconClasses} />
      </button>
    </div>
  );
}
