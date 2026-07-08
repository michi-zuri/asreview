import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Skeleton,
  Stack,
  TextField,
} from "@mui/material";
import React, { useContext } from "react";
import { useMutation, useQuery, useQueryClient } from "react-query";

import { ProjectAPI } from "api";
import { ProjectContext } from "context/ProjectContext";

const LlmSettingsCard = () => {
  const projectId = useContext(ProjectContext);
  const queryClient = useQueryClient();

  const { data: settings, isLoading } = useQuery(
    ["fetchLlmSettings", { project_id: projectId }],
    ProjectAPI.fetchLlmSettings,
    {
      enabled: !!projectId,
      refetchOnWindowFocus: false,
    },
  );

  const { mutate, isLoading: isSaving } = useMutation(
    ProjectAPI.mutateLlmSettings,
    {
      onSuccess: () => {
        queryClient.invalidateQueries(["fetchLlmSettings"]);
      },
    },
  );

  const [bufferSize, setBufferSize] = React.useState(20);
  const [maxConcurrent, setMaxConcurrent] = React.useState(3);
  const [staleHours, setStaleHours] = React.useState(24);
  const [criteriaText, setCriteriaText] = React.useState("");

  React.useEffect(() => {
    if (settings) {
      setBufferSize(settings.buffer_size ?? 20);
      setMaxConcurrent(settings.max_concurrent_llm ?? 3);
      setStaleHours(Math.round((settings.stale_timeout ?? 86400) / 3600));
      setCriteriaText(settings.criteria_text ?? "");
    }
  }, [settings]);

  const isPristine =
    settings &&
    bufferSize === (settings.buffer_size ?? 20) &&
    maxConcurrent === (settings.max_concurrent_llm ?? 3) &&
    staleHours === Math.round((settings.stale_timeout ?? 86400) / 3600) &&
    criteriaText === (settings.criteria_text ?? "");

  const handleSave = () => {
    mutate({
      project_id: projectId,
      buffer_size: bufferSize,
      max_concurrent_llm: maxConcurrent,
      stale_timeout: staleHours * 3600,
      criteria_text: criteriaText,
    });
  };

  return (
    <Card>
      <CardHeader
        title="LLM Screening"
        subheader="Full-text screening with AI"
      />
      <Divider />
      <CardContent>
        {isLoading ? (
          <Stack spacing={3}>
            <Skeleton variant="rectangular" height={56} />
            <Skeleton variant="rectangular" height={56} />
            <Skeleton variant="rectangular" height={56} />
            <Skeleton variant="rectangular" height={120} />
          </Stack>
        ) : (
          <Stack spacing={3}>
            <TextField
              label="Buffer size"
              type="number"
              value={bufferSize}
              onChange={(e) =>
                setBufferSize(Math.max(1, parseInt(e.target.value, 10) || 1))
              }
              helperText="Number of records to keep pre-screened in the dispatch queue"
              inputProps={{ min: 1 }}
              fullWidth
            />
            <TextField
              label="Max concurrent LLM workers"
              type="number"
              value={maxConcurrent}
              onChange={(e) =>
                setMaxConcurrent(Math.max(1, parseInt(e.target.value, 10) || 1))
              }
              helperText="Maximum parallel AI screening requests"
              inputProps={{ min: 1 }}
              fullWidth
            />
            <TextField
              label="Stale timeout (hours)"
              type="number"
              value={staleHours}
              onChange={(e) =>
                setStaleHours(Math.max(1, parseInt(e.target.value, 10) || 1))
              }
              helperText="Hours before a checked-out record is reassigned"
              inputProps={{ min: 1 }}
              fullWidth
            />
            <Box>
              <TextField
                label="Screening criteria"
                multiline
                minRows={4}
                value={criteriaText}
                onChange={(e) => setCriteriaText(e.target.value)}
                helperText="Edit with care — changing this changes the prompt and will cause not-yet-labeled records to be re-screened."
                fullWidth
              />
            </Box>
          </Stack>
        )}
      </CardContent>
      <Divider />
      <Box sx={{ display: "flex", justifyContent: "flex-end", p: 2 }}>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={isLoading || isPristine || isSaving}
        >
          {isSaving ? "Saving…" : "Save"}
        </Button>
      </Box>
    </Card>
  );
};

export default LlmSettingsCard;
