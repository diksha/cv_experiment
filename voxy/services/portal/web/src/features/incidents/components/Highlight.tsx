import React, { useState } from "react";
import { CircleWavyCheck } from "phosphor-react";
import classNames from "classnames";
import { useMutation } from "@apollo/client";
import { INCIDENT_HIGHLIGHT, INCIDENT_UNDO_HIGHLIGHT } from "features/incidents";
import { IncidentHighlight, IncidentHighlightVariables } from "__generated__/IncidentHighlight";
import { IncidentUndoHighlight, IncidentUndoHighlightVariables } from "__generated__/IncidentUndoHighlight";
import { LoadingButton } from "@mui/lab";
import { Dialog, Button } from "@mui/material";

interface HighlightProps {
  incidentId: string;
  highlighted: boolean;
}

export function Highlight(props: HighlightProps) {
  const [highlighted, setHighlighted] = useState(props.highlighted);
  const [modalOpen, setModalOpen] = useState(false);
  const [highlight, { loading: highlightLoading }] = useMutation<IncidentHighlight, IncidentHighlightVariables>(
    INCIDENT_HIGHLIGHT
  );
  const [undoHighlight, { loading: undoHighlightLoading }] = useMutation<
    IncidentUndoHighlight,
    IncidentUndoHighlightVariables
  >(INCIDENT_UNDO_HIGHLIGHT);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.nativeEvent.stopImmediatePropagation();
    setModalOpen(true);
  };

  const handleConfirm = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.nativeEvent.stopImmediatePropagation();

    const mutation = highlighted ? undoHighlight : highlight;
    mutation({ variables: { incidentId: props.incidentId } }).then(() => {
      setModalOpen(false);
      // Slight delay to prevent the modal content from shifting before its closed
      setTimeout(() => {
        setHighlighted(!highlighted);
      }, 500);
    });
  };

  const handleCloseModal = (e: React.MouseEvent) => {
    e.stopPropagation();
    setModalOpen(false);
  };

  const iconClasses = classNames(
    "h-6 w-6 md:h-8 md:w-8 transition-colors",
    highlighted ? "text-brand-purple-300 hover:text-purple-yellow-400" : "text-brand-gray-100 hover:text-brand-gray-200"
  );

  const loading = highlightLoading || undoHighlightLoading;
  const title = highlighted ? "Undo Highlight Incident" : "Highlight Incident";
  const confirmText = highlighted
    ? "This incident will no longer appear in highlighted incident lists."
    : "This is highly visible to customers so be sure this is something we want to draw attention to.";

  return (
    <div>
      <button
        data-ui-key={highlighted ? "button-highlight-incident" : "button-undo-highlight-incident"}
        onClick={handleClick}
        disabled={highlightLoading || undoHighlightLoading}
        title={highlighted ? "Un-highlight this incident." : "Highlight this incident"}
      >
        <CircleWavyCheck weight="fill" className={iconClasses} />
      </button>
      <Dialog open={modalOpen} onClose={() => setModalOpen(false)}>
        <div className="p-4 text-center">
          <div className="flex justify-center py-4">
            <CircleWavyCheck weight="fill" className="h-16 w-16 text-brand-purple-400" />
          </div>
          <div className="text-xl font-bold">{title}</div>
          <div className="pt-2 pb-8">
            <div>Are you sure?</div>
            <div>{confirmText}</div>
          </div>
          <div className="flex gap-4">
            <Button variant="outlined" onClick={handleCloseModal} className="flex-1" disabled={loading}>
              Cancel
            </Button>
            <LoadingButton variant="contained" onClick={handleConfirm} loading={loading} className="flex-1">
              Confirm
            </LoadingButton>
          </div>
        </div>
      </Dialog>
    </div>
  );
}
