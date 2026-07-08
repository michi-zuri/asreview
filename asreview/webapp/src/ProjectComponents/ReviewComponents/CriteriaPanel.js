import {
  Box,
  Drawer,
  IconButton,
  Skeleton,
  Stack,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import React from "react";
import { useQuery } from "react-query";

import { ProjectAPI } from "api";

const CriteriaPanel = ({ open, onClose, project_id }) => {
  const { data: settings, isLoading } = useQuery(
    ["fetchLlmSettings", { project_id }],
    ProjectAPI.fetchLlmSettings,
    {
      enabled: open && !!project_id,
      refetchOnWindowFocus: false,
    },
  );

  const criteriaText = settings?.criteria_text;

  return (
    <Drawer anchor="right" open={open} onClose={onClose}>
      <Box sx={{ width: 400, maxWidth: "90vw", p: 3 }}>
        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="center"
          sx={{ mb: 2 }}
        >
          <Typography variant="h6">Screening criteria</Typography>
          <IconButton
            onClick={onClose}
            size="small"
            aria-label="close criteria"
          >
            <CloseIcon />
          </IconButton>
        </Stack>
        {isLoading ? (
          <Stack spacing={1}>
            <Skeleton />
            <Skeleton />
            <Skeleton width="60%" />
          </Stack>
        ) : criteriaText ? (
          <Typography sx={{ whiteSpace: "pre-wrap" }}>
            {criteriaText}
          </Typography>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No screening criteria have been configured yet.
          </Typography>
        )}
      </Box>
    </Drawer>
  );
};

export default CriteriaPanel;
