import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Skeleton,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import React, { useContext } from "react";
import { useMutation, useQuery, useQueryClient } from "react-query";

import { ProjectAPI } from "api";
import { ProjectContext } from "context/ProjectContext";

const EMPTY_SETTINGS = {
  buffer_size: 20,
  max_concurrent_llm: 3,
  stale_timeout: 86400,
  criteria_text: "",
  api_key: "",
};

const LlmSettingsEditDialog = ({ open, onClose, initialSettings, onSave }) => {
  const [state, setState] = React.useState(EMPTY_SETTINGS);

  React.useEffect(() => {
    if (open && initialSettings) {
      setState({
        buffer_size: initialSettings.buffer_size ?? 20,
        max_concurrent_llm: initialSettings.max_concurrent_llm ?? 3,
        stale_hours: Math.round(
          (initialSettings.stale_timeout ?? 86400) / 3600,
        ),
        criteria_text: initialSettings.criteria_text ?? "",
        api_key: initialSettings.api_key ?? "",
      });
    }
  }, [open, initialSettings]);

  const handleNumberChange = (field) => (e) => {
    setState((prev) => ({
      ...prev,
      [field]: Math.max(1, parseInt(e.target.value, 10) || 1),
    }));
  };

  const handleTextChange = (field) => (e) => {
    setState((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const handleSave = () => {
    onSave({
      buffer_size: state.buffer_size,
      max_concurrent_llm: state.max_concurrent_llm,
      stale_timeout: state.stale_hours * 3600,
      criteria_text: state.criteria_text,
      api_key: state.api_key.trim(),
    });
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>LLM screening settings</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ pt: 1 }}>
          <TextField
            label="Buffer size"
            type="number"
            value={state.buffer_size}
            onChange={handleNumberChange("buffer_size")}
            helperText="Number of records to keep pre-screened in the dispatch queue"
            inputProps={{ min: 1 }}
            fullWidth
          />
          <TextField
            label="Max concurrent LLM workers"
            type="number"
            value={state.max_concurrent_llm}
            onChange={handleNumberChange("max_concurrent_llm")}
            helperText="Maximum parallel AI screening requests"
            inputProps={{ min: 1 }}
            fullWidth
          />
          <TextField
            label="Stale timeout (hours)"
            type="number"
            value={state.stale_hours}
            onChange={handleNumberChange("stale_hours")}
            helperText="Hours before a checked-out record is reassigned"
            inputProps={{ min: 1 }}
            fullWidth
          />
          <TextField
            label="Screening criteria"
            multiline
            minRows={4}
            value={state.criteria_text}
            onChange={handleTextChange("criteria_text")}
            helperText="Changing this will cause not-yet-labeled records to be re-screened."
            fullWidth
          />
          <TextField
            label="Anthropic API key"
            type="password"
            value={state.api_key}
            onChange={handleTextChange("api_key")}
            helperText="Leave blank to use the ANTHROPIC_API_KEY environment variable"
            fullWidth
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained">
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

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

  const [editDialogOpen, setEditDialogOpen] = React.useState(false);

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
          <Stack spacing={2}>
            <Box>
              <Typography variant="subtitle2">Buffer size</Typography>
              <Typography variant="body2" color="text.secondary">
                {settings?.buffer_size ?? 20} — records kept pre-screened in the
                dispatch queue
              </Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2">
                Max concurrent LLM workers
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {settings?.max_concurrent_llm ?? 3} — maximum parallel AI
                screening requests
              </Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2">Stale timeout</Typography>
              <Typography variant="body2" color="text.secondary">
                {Math.round((settings?.stale_timeout ?? 86400) / 3600)} hours
                before a checked-out record is reassigned
              </Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2">Screening criteria</Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ whiteSpace: "pre-wrap" }}
              >
                {settings?.criteria_text || "(none set)"}
              </Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2">Anthropic API key</Typography>
              <Typography variant="body2" color="text.secondary">
                {settings?.api_key
                  ? "********"
                  : "(using ANTHROPIC_API_KEY env var)"}
              </Typography>
            </Box>
          </Stack>
        )}
      </CardContent>
      <Divider />
      <Box sx={{ display: "flex", justifyContent: "flex-end", p: 2 }}>
        <Button
          variant="contained"
          startIcon={<EditIcon />}
          onClick={() => setEditDialogOpen(true)}
          disabled={isLoading || isSaving}
        >
          Edit settings
        </Button>
      </Box>
      <LlmSettingsEditDialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        initialSettings={settings}
        onSave={(newSettings) => {
          mutate({ project_id: projectId, ...newSettings });
          setEditDialogOpen(false);
        }}
      />
    </Card>
  );
};

export default LlmSettingsCard;
