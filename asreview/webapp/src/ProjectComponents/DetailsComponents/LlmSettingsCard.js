import {
  Alert,
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
  FormControl,
  FormHelperText,
  InputLabel,
  MenuItem,
  Select,
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
  criteria_text: "",
  api_key: "",
  model: "claude-opus-4-6",
};

const LlmSettingsEditDialog = ({
  open,
  onClose,
  initialSettings,
  onSave,
  saveError,
  isSaving,
  projectId,
}) => {
  const [state, setState] = React.useState(EMPTY_SETTINGS);
  const [apiKeyTouched, setApiKeyTouched] = React.useState(false);
  const [models, setModels] = React.useState([]);
  const [modelsLoading, setModelsLoading] = React.useState(false);
  const [modelsFailed, setModelsFailed] = React.useState(false);
  const modelsRequest = React.useRef(0);

  // Fetch the live model listing. apiKey undefined -> the project's
  // stored key is used; otherwise the given key (never stored).
  const fetchModels = React.useCallback(
    (apiKey) => {
      const requestId = ++modelsRequest.current;
      setModelsLoading(true);
      setModelsFailed(false);
      ProjectAPI.fetchLlmModels(
        apiKey === undefined
          ? { project_id: projectId }
          : { project_id: projectId, api_key: apiKey },
      )
        .then((list) => {
          if (requestId === modelsRequest.current) {
            setModels(list);
            setModelsLoading(false);
          }
        })
        .catch(() => {
          if (requestId === modelsRequest.current) {
            setModelsFailed(true);
            setModelsLoading(false);
          }
        });
    },
    [projectId],
  );

  React.useEffect(() => {
    if (open && initialSettings) {
      // If api_key is true (masked boolean), the backend has a key but we
      // don't know its value — show an empty field.
      const rawKey = initialSettings.api_key;
      setState({
        buffer_size: initialSettings.buffer_size ?? 20,
        max_concurrent_llm: initialSettings.max_concurrent_llm ?? 3,
        criteria_text: initialSettings.criteria_text ?? "",
        api_key: typeof rawKey === "string" ? rawKey : "",
        model: initialSettings.model ?? "claude-opus-4-6",
      });
      setApiKeyTouched(false);
      setModels(initialSettings.models ?? []);
      setModelsFailed(false);
      setModelsLoading(false);
      // Refresh the listing every time the dialog opens; with a stored
      // key the live catalog is fetched.
      if (initialSettings.api_key) {
        fetchModels(undefined);
      } else {
        // Invalidate any request still in flight from a previous open.
        modelsRequest.current += 1;
      }
    }
  }, [open, initialSettings, fetchModels]);

  // Re-fetch the listing (debounced) as the user types an API key.
  React.useEffect(() => {
    if (!open || !apiKeyTouched) {
      return undefined;
    }
    const timer = setTimeout(() => {
      fetchModels(state.api_key.trim());
    }, 500);
    return () => clearTimeout(timer);
  }, [open, apiKeyTouched, state.api_key, fetchModels]);

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
    const payload = {
      buffer_size: state.buffer_size,
      max_concurrent_llm: state.max_concurrent_llm,
      criteria_text: state.criteria_text,
      model: state.model,
    };
    // Only send the API key if the user actually touched the field.
    if (apiKeyTouched) {
      payload.api_key = state.api_key.trim();
    }
    onSave(payload);
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>LLM screening settings</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ pt: 1 }}>
          <FormControl fullWidth>
            <InputLabel id="llm-model-label">Model</InputLabel>
            <Select
              labelId="llm-model-label"
              label="Model"
              value={state.model}
              onChange={(e) => {
                setState((prev) => ({ ...prev, model: e.target.value }));
              }}
            >
              {models.map((m) => (
                <MenuItem key={m.id} value={m.id}>
                  {m.label}
                </MenuItem>
              ))}
              {state.model && !models.some((m) => m.id === state.model) && (
                <MenuItem key={state.model} value={state.model}>
                  {state.model}
                </MenuItem>
              )}
            </Select>
            <FormHelperText>
              {modelsLoading
                ? "Loading available models…"
                : modelsFailed
                  ? "Could not load the model list; showing the last known options."
                  : "Changing the model invalidates pre-processed records without a final decision."}
            </FormHelperText>
          </FormControl>
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
            onChange={(e) => {
              setApiKeyTouched(true);
              handleTextChange("api_key")(e);
            }}
            helperText="Required to enable LLM screening"
            fullWidth
          />
        </Stack>
        {saveError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {saveError}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" disabled={isSaving}>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const LlmSettingsCard = () => {
  const projectId = useContext(ProjectContext);
  const queryClient = useQueryClient();
  const [editDialogOpen, setEditDialogOpen] = React.useState(false);
  const [saveError, setSaveError] = React.useState(null);

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
        setSaveError(null);
        setEditDialogOpen(false);
      },
      onError: (error) => {
        // Keep the dialog open and show the backend's message (e.g. the
        // model id failed live validation).
        setSaveError(
          error?.message || "Could not save the LLM screening settings.",
        );
      },
    },
  );

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
              <Typography variant="subtitle2">Model</Typography>
              <Typography variant="body2" color="text.secondary">
                {settings?.models?.find((m) => m.id === settings?.model)
                  ?.label ??
                  settings?.model ??
                  "claude-opus-4-6"}
              </Typography>
            </Box>
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
                  : "(no API key provided, LLM processing is disabled)"}
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
          onClick={() => {
            setSaveError(null);
            setEditDialogOpen(true);
          }}
          disabled={isLoading || isSaving}
        >
          Edit settings
        </Button>
      </Box>
      <LlmSettingsEditDialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        initialSettings={settings}
        projectId={projectId}
        saveError={saveError}
        isSaving={isSaving}
        onSave={(newSettings) => {
          // The dialog stays open on failure so the error is visible.
          mutate({ project_id: projectId, ...newSettings });
        }}
      />
    </Card>
  );
};

export default LlmSettingsCard;
