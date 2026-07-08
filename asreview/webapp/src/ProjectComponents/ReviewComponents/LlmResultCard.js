import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import PictureAsPdfOffIcon from "@mui/icons-material/PictureAsPdf";
import {
  Box,
  Card,
  CardContent,
  CircularProgress,
  Stack,
  Typography,
} from "@mui/material";
import React from "react";
import { useQuery } from "react-query";

import { ProjectAPI } from "api";

const LlmResultCard = ({ project_id, record_id, llm }) => {
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

  if (!llm) return null;

  const meta = data || llm;

  const renderContent = () => {
    switch (meta.status) {
      case "queued":
      case "in_flight":
        return (
          <Stack direction="row" spacing={1.5} alignItems="center">
            <CircularProgress size={18} />
            <Typography variant="body2" color="text.secondary">
              Analyzing full text…
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
                Full-text screening complete
              </Typography>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              {meta.model}
              {latency !== null && ` · ${latency}s`}
              {meta.input_tokens != null &&
                ` · ${meta.input_tokens} in / ${meta.output_tokens} out tokens`}
              {dispatchedTime && ` · dispatched ${dispatchedTime}`}
            </Typography>
          </Stack>
        );

      case "failed":
        return (
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
        );

      case "missing_pdf":
        return (
          <Stack direction="row" spacing={1} alignItems="center">
            <PictureAsPdfOffIcon fontSize="small" color="disabled" />
            <Typography variant="body2" color="text.secondary">
              No PDF available for this record.
            </Typography>
          </Stack>
        );

      default:
        return null;
    }
  };

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
        {renderContent()}
      </CardContent>
    </Card>
  );
};

export default LlmResultCard;
