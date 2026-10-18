import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import PictureAsPdfOffIcon from "@mui/icons-material/PictureAsPdf";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  Alert,
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

// Statuses in which no further LLM work is scheduled for the record.
const TERMINAL_STATUSES = ["ready", "failed", "missing_pdf"];

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

  const { data } = useQuery(
    ["fetchLlmMeta", { project_id, record_id }],
    ProjectAPI.fetchLlmMeta,
    {
      initialData: llm,
      enabled: !!llm,
      refetchOnWindowFocus: false,
      // Poll only while the latest known status is non-terminal. Deriving
      // the interval from the query data (instead of the `llm` prop) means
      // polling resumes when an invalidation brings in a queued/in_flight
      // status — e.g. after a PDF upload or replace re-queued the record.
      refetchInterval: (current) =>
        current && TERMINAL_STATUSES.includes(current.status) ? false : 4000,
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

  // Auto-apply for records that are already ready when first mounted.
  const didAutoApply = React.useRef(false);
  React.useEffect(() => {
    didAutoApply.current = false;
  }, [record_id]);

  React.useEffect(() => {
    if (
      meta?.status === "ready" &&
      !isDirty &&
      !didAutoApply.current &&
      onApplyLlm
    ) {
      didAutoApply.current = true;
      onApplyLlm();
    }
  }, [meta?.status, isDirty, onApplyLlm]);

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
        const pdfMismatch =
          meta.pdf_match?.plausible === false ? meta.pdf_match : null;
        return (
          <Stack spacing={0.5}>
            {pdfMismatch && (
              <Alert severity="warning">
                <Typography variant="body2" fontWeight="medium">
                  The PDF may not belong to this article
                </Typography>
                <Typography variant="body2">
                  {pdfMismatch.issue ||
                    "The LLM could not match the PDF to this article's title and abstract."}
                </Typography>
                <Typography variant="body2">
                  No suggestions were pre-filled. Check the PDF attachment and
                  replace it if it is wrong.
                </Typography>
              </Alert>
            )}
            {!pdfMismatch && (
              <Stack direction="row" spacing={1} alignItems="center">
                <CheckCircleOutlineIcon fontSize="small" color="success" />
                <Typography variant="body2" fontWeight="medium">
                  {justLanded
                    ? "An automated LLM appraisal just became available. You can apply it with the button below."
                    : "Pre-processing of full-text by LLM complete, please review automated suggestions with care."}
                </Typography>
              </Stack>
            )}
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
            <Typography variant="caption" color="text.secondary">
              Upload the PDF with the PDF button on this card. Full-text
              screening starts automatically after the upload.
            </Typography>
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
