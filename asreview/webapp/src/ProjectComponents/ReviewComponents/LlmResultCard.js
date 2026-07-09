import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import PictureAsPdfOffIcon from "@mui/icons-material/PictureAsPdf";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
import React from "react";
import { useQuery, useQueryClient } from "react-query";

import { ProjectAPI } from "api";

const LlmResultCard = ({
  project_id,
  record_id,
  llm,
  isDirty = false,
  onApplyLlm = null,
}) => {
  const queryClient = useQueryClient();
  const [applyDialogOpen, setApplyDialogOpen] = React.useState(false);
  const [busy, setBusy] = React.useState(false);

  const isTerminal =
    llm &&
    (llm.status === "ready" ||
      llm.status === "failed" ||
      llm.status === "missing_pdf");

  const { data } = useQuery(
    ["fetchLlmMeta", { project_id, record_id }],
    ProjectAPI.fetchLlmMeta,
    {
      initialData: llm,
      enabled: !!llm,
      refetchInterval: isTerminal ? false : 4000,
      refetchOnWindowFocus: false,
    },
  );

  const handleReprocess = () => {
    setBusy(true);
    ProjectAPI.reprocessRecord({ project_id, record_id })
      .then(() => {
        queryClient.invalidateQueries({
          queryKey: ["fetchLlmMeta", { project_id, record_id }],
        });
      })
      .finally(() => setBusy(false));
  };

  const handleRecheckPdf = () => {
    setBusy(true);
    ProjectAPI.recheckPdf({ project_id, record_id })
      .then(() => {
        queryClient.invalidateQueries({
          queryKey: ["fetchLlmMeta", { project_id, record_id }],
        });
      })
      .finally(() => setBusy(false));
  };

  const handleApplyClick = () => {
    if (isDirty) {
      setApplyDialogOpen(true);
    } else {
      onApplyLlm && onApplyLlm();
    }
  };

  const handleApplyConfirm = () => {
    setApplyDialogOpen(false);
    onApplyLlm && onApplyLlm();
  };

  const meta = data || llm;

  // Detect transition from queued/in_flight → ready.
  const prevStatusRef = React.useRef(meta?.status);
  const [justLanded, setJustLanded] = React.useState(false);
  React.useEffect(() => {
    if (!meta) return;
    const prev = prevStatusRef.current;
    prevStatusRef.current = meta.status;
    if (
      meta.status === "ready" &&
      (prev === "queued" || prev === "in_flight")
    ) {
      if (!isDirty && onApplyLlm) {
        onApplyLlm();
      } else if (isDirty) {
        setJustLanded(true);
      }
    }
  }, [meta, isDirty, onApplyLlm]);

  if (!llm) return null;

  const renderContent = () => {
    switch (meta.status) {
      case "queued":
        return (
          <Typography variant="body2" color="text.secondary">
            Full-text analysis by LLM is queued, but not processing yet. You
            should proceed with manual screening for now.
          </Typography>
        );

      case "in_flight":
        return (
          <Stack direction="row" spacing={1.5} alignItems="center">
            <CircularProgress size={18} />
            <Typography variant="body2" color="text.secondary">
              Analyzing full text...
            </Typography>
          </Stack>
        );

      case "ready":
        const latency =
          meta.created_at && meta.dispatched_at
            ? Math.round(meta.created_at - meta.dispatched_at)
            : null;
        const dispatchedTime = meta.dispatched_at
          ? new Date(meta.dispatched_at * 1000).toLocaleString()
          : null;
        return (
          <Stack spacing={0.5}>
            <Stack direction="row" spacing={1} alignItems="center">
              <CheckCircleOutlineIcon fontSize="small" color="success" />
              <Typography variant="body2" fontWeight="medium">
                {justLanded
                  ? "An automated LLM appraisal just became available. You can apply it with the button below."
                  : "Pre-processing of full-text by LLM complete, please review automated suggestions with care."}
              </Typography>
            </Stack>
            <Stack
              direction="row"
              spacing={1}
              alignItems="center"
              justifyContent="space-between"
            >
              <Typography variant="caption" color="text.secondary">
                {meta.model}
                {latency !== null && ` \u00b7 ${latency}s`}
                {meta.input_tokens != null &&
                  ` \u00b7 ${meta.input_tokens} in / ${meta.output_tokens} out tokens`}
                {dispatchedTime && ` \u00b7 dispatched ${dispatchedTime}`}
              </Typography>
              {onApplyLlm && (
                <Tooltip
                  title={
                    !isDirty
                      ? "LLM suggestions are already applied to the form below"
                      : ""
                  }
                >
                  <span>
                    <Button
                      size="small"
                      startIcon={<AutoFixHighIcon />}
                      disabled={!isDirty}
                      onClick={handleApplyClick}
                    >
                      Apply suggestions
                    </Button>
                  </span>
                </Tooltip>
              )}
            </Stack>
          </Stack>
        );

      case "failed":
        return (
          <Stack spacing={0.5}>
            <Stack direction="row" spacing={1} alignItems="flex-start">
              <ErrorOutlineIcon fontSize="small" color="error" />
              <Box>
                <Typography variant="body2" fontWeight="medium">
                  Full-text screening failed
                </Typography>
                {meta.last_error && (
                  <Typography variant="caption" color="text.secondary">
                    {meta.last_error}
                  </Typography>
                )}
              </Box>
            </Stack>
            <Stack direction="row" justifyContent="flex-end">
              <Button
                size="small"
                startIcon={<RefreshIcon />}
                disabled={busy}
                onClick={handleReprocess}
              >
                Re-screen
              </Button>
            </Stack>
          </Stack>
        );

      case "missing_pdf":
        return (
          <Stack spacing={0.5}>
            <Stack direction="row" spacing={1} alignItems="center">
              <PictureAsPdfOffIcon fontSize="small" color="disabled" />
              <Typography variant="body2" color="text.secondary">
                No PDF available for this record.
              </Typography>
            </Stack>
            <Stack direction="row" justifyContent="flex-end">
              <Button
                size="small"
                startIcon={<RefreshIcon />}
                disabled={busy}
                onClick={handleRecheckPdf}
              >
                Recheck PDF
              </Button>
            </Stack>
          </Stack>
        );

      default:
        return null;
    }
  };

  return (
    <>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
          {renderContent()}
        </CardContent>
      </Card>
      <Dialog open={applyDialogOpen} onClose={() => setApplyDialogOpen(false)}>
        <DialogTitle>Apply LLM suggestions</DialogTitle>
        <DialogContent>
          <DialogContentText>
            You have made changes to the screening form. Applying the LLM
            suggestions will discard your current input and overwrite it with
            the automated appraisal. Continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApplyDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleApplyConfirm} variant="contained">
            Apply
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default LlmResultCard;
