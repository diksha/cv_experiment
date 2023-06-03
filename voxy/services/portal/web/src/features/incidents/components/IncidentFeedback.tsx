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
import React, { SyntheticEvent, useRef, useState } from "react";
import { Prohibit, Check, ChatText, Question, ThumbsDown, ThumbsUp } from "phosphor-react";
import { IconRadioGroup, Modal, ModalBody } from "ui";
import { Button } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { CREATE_INCIDENT_FEEDBACK } from "features/incidents";
import { toTitleCase } from "shared/utilities/strings";
import { useMutation } from "@apollo/client";
import { CreateIncidentFeedback, CreateIncidentFeedbackVariables } from "__generated__/CreateIncidentFeedback";

interface FeedbackProps {
  incidentId: string;
  incidentTitle?: string;
  incidentOrganization?: string;
  fullWidth?: boolean;
  onSubmit?: () => void;
  onClose?: (e: SyntheticEvent) => void;
  timestampOfMostRecentSubmit?: number;
}

export function IncidentFeedback(props: FeedbackProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [selectedOption, setSelectedOption] = useState<string>();
  const [submitted, setSubmitted] = useState(false);
  const [text, setText] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const [addFeedback, { loading }] = useMutation<CreateIncidentFeedback, CreateIncidentFeedbackVariables>(
    CREATE_INCIDENT_FEEDBACK
  );

  const handleSubmit = (e: SyntheticEvent) => {
    e.stopPropagation();
    if (selectedOption && !submitted && !loading) {
      setErrorMsg("");

      addFeedback({
        variables: {
          incidentId: props.incidentId,
          feedbackType: "incident_accuracy",
          feedbackValue: selectedOption,
          feedbackText: text,
          // conver to seconds
          incidentServedTimestampSeconds:
            props.timestampOfMostRecentSubmit && Math.round(props.timestampOfMostRecentSubmit / 1000),
          elapsedMillisecondsBetweenReviews: props.timestampOfMostRecentSubmit
            ? Date.now() - props.timestampOfMostRecentSubmit
            : null,
        },
      })
        .then((data) => {
          if (data?.data?.createIncidentFeedback?.userErrors?.length) {
            setSubmitted(false);
            setErrorMsg("You don't have permission to create incident feedback.");
            return;
          }
          setSubmitted(true);
          setSelectedOption("");
          setText("");
          props.onSubmit && props.onSubmit();
        })
        .catch(() => {
          setSubmitted(false);
          setErrorMsg("Something went wrong when submitting your feedback.");
        });
    }
  };

  return (
    <div>
      {!submitted && (
        <>
          {props.incidentTitle ? (
            <div className="text-2xl pb-2">{props.incidentTitle && toTitleCase(props.incidentTitle)}</div>
          ) : null}
          {props.incidentOrganization ? (
            <div className="pb-2">({props.incidentOrganization && toTitleCase(props.incidentOrganization)})</div>
          ) : null}
          <IconRadioGroup
            label="Incident feedback"
            selected={selectedOption}
            onChange={(option: string) => setSelectedOption(option)}
            options={[
              {
                name: "Valid",
                value: "valid",
                description: "This event was identified correctly",
                icon: <ThumbsUp size={24} />,
                iconClass: "text-green-200",
                selectedIconClass: "text-green-700",
                uiKey: "radio-button-valid-incident",
              },
              {
                name: "Invalid",
                value: "invalid",
                description: "This event was NOT identified correctly",
                icon: <ThumbsDown size={24} />,
                iconClass: "text-red-200",
                selectedIconClass: "text-red-700",
                uiKey: "radio-button-not-valid-incident",
              },
              {
                name: "Unsure",
                value: "unsure",
                description: "I'm not sure if this event was identifed correctly",
                icon: <Question size={24} />,
                iconClass: "text-gray-300",
                selectedIconClass: "text-gray-700",
                uiKey: "radio-button-not-unsure-incident",
              },
              {
                name: "Corrupted",
                value: "corrupt",
                description: "The video is corrupted",
                icon: <Prohibit size={24} />,
                iconClass: "text-gray-300",
                selectedIconClass: "text-gray-700",
                uiKey: "radio-button-corrupted-incident",
              },
            ]}
          />
          {selectedOption || props.onClose ? (
            <>
              <div className="mt-4">
                <textarea
                  data-ui-key="textarea-provide-feedback-details"
                  value={text}
                  className="w-full rounded-md border-gray-300"
                  ref={textareaRef}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Provide additional details here..."
                />
              </div>
              <div className="grid grid-cols-2 gap-4 my-2">
                {errorMsg && <div>{errorMsg}</div>}
                {props.onClose && (
                  <Button
                    data-ui-key="button-provide-feedback-cancel"
                    variant="outlined"
                    disabled={loading}
                    onClick={props.onClose}
                  >
                    Cancel
                  </Button>
                )}
                <LoadingButton
                  data-ui-key="button-provide-feedback-submit"
                  variant="contained"
                  loading={loading}
                  disabled={!selectedOption || loading || submitted}
                  onClick={handleSubmit}
                >
                  Submit
                </LoadingButton>
              </div>
            </>
          ) : null}
        </>
      )}
      {submitted && (
        <div className="grid content-center text-center h-full py-16 px-4">
          <div>
            <Check size={48} className="text-green-400 mx-auto" />
            <div>Your feedback was received!</div>
          </div>
        </div>
      )}
    </div>
  );
}

export function FeedbackButton(props: FeedbackProps) {
  const [active, setActive] = useState(false);

  const handleActivate = (e: any) => {
    e.stopPropagation();
    setActive(true);
  };

  const handleClose = () => {
    setActive(false);
  };

  const autoClose = () => {
    setTimeout(() => setActive(false), 1000);
  };

  return (
    <>
      <Button variant="outlined" onClick={handleActivate} startIcon={<ChatText />} fullWidth={props.fullWidth}>
        Provide Feedback
      </Button>
      <Modal open={active} onClose={() => handleClose()}>
        <ModalBody>
          <IncidentFeedback {...props} onClose={handleClose} onSubmit={autoClose} />
        </ModalBody>
      </Modal>
    </>
  );
}
