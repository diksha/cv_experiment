import { MouseEvent } from "react";
import { Check, ArrowUUpLeft } from "phosphor-react";
import { RESOLVE_INCIDENT, REOPEN_INCIDENT } from "features/incidents";
import { LoadingButton } from "@mui/lab";
import { useMutation } from "@apollo/client";
import { IncidentResolve, IncidentResolveVariables } from "__generated__/IncidentResolve";
import { IncidentReopen, IncidentReopenVariables } from "__generated__/IncidentReopen";

interface ResolveButtonProps {
  incidentId: string;
  resolved: boolean;
  fullWidth?: boolean;
  onResolve: () => void;
  onReopen: () => void;
}

export function ResolveButton({ incidentId, resolved, fullWidth, onResolve, onReopen }: ResolveButtonProps) {
  const [resolveIncident, { loading: resolveLoading }] = useMutation<IncidentResolve, IncidentResolveVariables>(
    RESOLVE_INCIDENT,
    {
      onCompleted: () => {
        onResolve();
      },
    }
  );
  const [reopenIncident, { loading: reopenLoading }] = useMutation<IncidentReopen, IncidentReopenVariables>(
    REOPEN_INCIDENT,
    {
      onCompleted: () => {
        onReopen();
      },
    }
  );
  const handleReopenClick = (_: MouseEvent<HTMLButtonElement>) => {
    reopenIncident({
      variables: {
        incidentId,
      },
    });
  };

  const handleResolveClick = (_: MouseEvent<HTMLButtonElement>) => {
    resolveIncident({
      variables: {
        incidentId,
      },
    });
  };
  return (
    <>
      {resolved ? (
        <LoadingButton
          id={`reopen-incident-button-${incidentId}`}
          data-ui-key="button-re-open-incident"
          variant="outlined"
          onClick={handleReopenClick}
          startIcon={<ArrowUUpLeft />}
          loading={reopenLoading}
          loadingPosition="start"
          fullWidth={fullWidth}
        >
          Re-open Incident
        </LoadingButton>
      ) : (
        <LoadingButton
          variant="contained"
          id={`resolve-incident-button-${incidentId}`}
          data-ui-key="button-resolve-incident"
          onClick={handleResolveClick}
          startIcon={<Check />}
          loading={resolveLoading}
          loadingPosition="start"
          fullWidth={fullWidth}
        >
          Resolve Incident
        </LoadingButton>
      )}
    </>
  );
}
