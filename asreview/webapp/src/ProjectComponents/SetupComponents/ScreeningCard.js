import {
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  IconButton,
  Popover,
  Skeleton,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import * as React from "react";
import { useContext } from "react";
import { useMutation, useQuery, useQueryClient } from "react-query";

import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import { ProjectAPI } from "api";
import { ProjectContext } from "context/ProjectContext";
import { LoadingCardHeader } from "StyledComponents/LoadingCardheader";
import { StyledLightBulb } from "StyledComponents/StyledLightBulb";

const EMPTY_ZOTERO_CONFIG = {
  api_key: "",
  group_id: "",
  group_slug: "",
};

const ZoteroEditDialog = ({
  open,
  onClose,
  initialConfig,
  onSave,
  onDelete,
  projectId,
}) => {
  const [state, setState] = React.useState(EMPTY_ZOTERO_CONFIG);
  const [apiKeyTouched, setApiKeyTouched] = React.useState(false);
  const [validating, setValidating] = React.useState(false);
  const [validationError, setValidationError] = React.useState("");

  React.useEffect(() => {
    if (open) {
      if (initialConfig) {
        // If api_key is true (masked), we know a key exists but not its
        // value — show an empty password field.
        const rawKey = initialConfig.api_key;
        setState({
          api_key: typeof rawKey === "string" ? rawKey : "",
          group_id: initialConfig.group_id || "",
        });
      } else {
        setState(structuredClone(EMPTY_ZOTERO_CONFIG));
      }
      setApiKeyTouched(false);
      setValidationError("");
    }
  }, [open, initialConfig]);

  const handleChange = (field) => (e) => {
    setState((prev) => ({ ...prev, [field]: e.target.value }));
    setValidationError("");
  };

  const handleSave = async () => {
    const apiKey = state.api_key.trim();
    const groupId = state.group_id.trim();

    let slug = initialConfig?.group_slug || "";

    // If the user provided a new API key, validate it against Zotero.
    // The group name from the API becomes the slug.
    if (apiKeyTouched && apiKey && groupId) {
      setValidating(true);
      setValidationError("");
      try {
        const result = await ProjectAPI.validateZotero({
          project_id: projectId,
          group_id: groupId,
          api_key: apiKey,
        });
        slug = result.name || "";
      } catch (err) {
        setValidationError(
          err?.message || "Could not validate Zotero credentials.",
        );
        setValidating(false);
        return;
      }
      setValidating(false);
    }

    const payload = {
      group_id: groupId,
      group_slug: slug,
    };
    // Send true (masked sentinel) to preserve the existing key unless the
    // user actually edited the field.
    payload.api_key = apiKeyTouched ? apiKey : true;
    onSave(payload);
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Zotero credentials</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ pt: 1 }}>
          <TextField
            fullWidth
            id="zotero-api-key"
            label="API Key"
            type="password"
            value={state.api_key}
            onChange={(e) => {
              setApiKeyTouched(true);
              handleChange("api_key")(e);
            }}
            helperText="Zotero API key with read access to the group library"
          />
          <TextField
            fullWidth
            id="zotero-group-id"
            label="Group ID"
            value={state.group_id}
            onChange={handleChange("group_id")}
            helperText="Numeric ID of the Zotero group library"
          />
          {validationError && (
            <Typography variant="body2" color="error">
              {validationError}
            </Typography>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        {onDelete && (
          <Button
            onClick={() => {
              onDelete();
              onClose();
            }}
            color="error"
            startIcon={<DeleteIcon />}
            sx={{ mr: "auto" }}
          >
            Delete credentials
          </Button>
        )}
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" disabled={validating}>
          {validating ? "Validating..." : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const InfoPopover = ({ anchorEl, handlePopoverClose }) => {
  return (
    <Popover
      open={Boolean(anchorEl)}
      anchorEl={anchorEl}
      onClose={handlePopoverClose}
      anchorOrigin={{
        vertical: "bottom",
        horizontal: "right",
      }}
      transformOrigin={{
        vertical: "top",
        horizontal: "right",
      }}
      PaperProps={{
        sx: {
          borderRadius: 3,
          maxWidth: 350,
        },
      }}
    >
      <Box
        sx={(theme) => ({
          p: 3,
          maxHeight: "80vh",
          overflow: "auto",
          "&::-webkit-scrollbar": {
            width: "8px",
            background: "transparent",
          },
          "&::-webkit-scrollbar-thumb": {
            background: theme.palette.grey[300],
            borderRadius: "4px",
            "&:hover": {
              background: theme.palette.grey[400],
            },
          },
          "&::-webkit-scrollbar-track": {
            background: "transparent",
            borderRadius: "4px",
          },
          scrollbarWidth: "thin",
          scrollbarColor: `${theme.palette.grey[300]} transparent`,
        })}
      >
        <Stack spacing={2}>
          <Box>
            <Typography variant="h6" sx={{ mb: 1 }}>
              Screening Options
            </Typography>
            <Typography variant="body2" align="justify">
              Configure how records are presented during the screening process.
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
              Reassign stale records
            </Typography>
            <Typography variant="body2" color="text.secondary" align="justify">
              When enabled, records that have been checked out by a reviewer for
              more than 24 hours are automatically reassigned to the dispatch
              queue so another reviewer can screen them.
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
              Hide DOI and URL links
            </Typography>
            <Typography variant="body2" color="text.secondary" align="justify">
              When enabled, the DOI and URL buttons are hidden from the
              screening interface. This prevents reviewers from accessing the
              full text of a record, which is useful for validation studies
              where screening decisions should be based solely on the title and
              abstract.
            </Typography>
          </Box>
        </Stack>
      </Box>
    </Popover>
  );
};

const ScreeningCard = () => {
  const project_id = useContext(ProjectContext);
  const queryClient = useQueryClient();
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [zoteroEditOpen, setZoteroEditOpen] = React.useState(false);

  const { data, isLoading } = useQuery(
    ["fetchProject", { project_id: project_id }],
    ProjectAPI.fetchInfo,
    {
      refetchOnWindowFocus: false,
    },
  );

  const { data: zoteroData, isLoading: zoteroLoading } = useQuery(
    ["fetchZoteroConfig", { project_id }],
    ProjectAPI.fetchZoteroConfig,
    {
      enabled: !!project_id,
      refetchOnWindowFocus: false,
    },
  );

  const { mutate } = useMutation(ProjectAPI.mutateInfo, {
    onSuccess: () => {
      queryClient.invalidateQueries(["fetchProject", { project_id }]);
    },
  });

  const { mutate: saveZotero } = useMutation(ProjectAPI.mutateZoteroConfig, {
    onSuccess: () => {
      queryClient.invalidateQueries(["fetchZoteroConfig", { project_id }]);
    },
    onError: (err) => console.error("Failed to save Zotero config:", err),
  });

  const hideLinks = data?.hide_links ?? false;
  const reassignStale = data?.reassign_stale ?? false;
  const zoteroConfig = zoteroData || EMPTY_ZOTERO_CONFIG;
  const zoteroConfigured = Boolean(
    zoteroConfig.api_key && zoteroConfig.group_id,
  );

  return (
    <Card>
      <LoadingCardHeader
        title="Screening"
        subheader="Configure the screening interface"
        isLoading={isLoading}
        action={
          <IconButton
            onClick={(event) => {
              setAnchorEl(event.currentTarget);
            }}
          >
            <StyledLightBulb />
          </IconButton>
        }
      />

      <InfoPopover
        anchorEl={anchorEl}
        handlePopoverClose={() => {
          setAnchorEl(null);
        }}
      />

      <CardContent>
        <FormControlLabel
          control={
            <Switch
              checked={reassignStale}
              onChange={(e) => {
                mutate({
                  project_id: project_id,
                  reassign_stale: e.target.checked,
                });
              }}
            />
          }
          label="Reassign stale records after 24 hours"
        />
        <br />
        <FormControlLabel
          control={
            <Switch
              checked={hideLinks}
              onChange={(e) => {
                mutate({
                  project_id: project_id,
                  hide_links: e.target.checked,
                });
              }}
            />
          }
          label="Hide DOI and URL links during screening"
        />
      </CardContent>

      <Divider />

      <CardContent>
        <Stack spacing={1}>
          <Stack
            direction="row"
            spacing={1}
            alignItems="center"
            justifyContent="space-between"
          >
            <Typography variant="subtitle1">
              Config for Zotero PDF links
            </Typography>
            <Tooltip
              title={
                <Stack spacing={1} sx={{ p: 0.5 }}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <PictureAsPdfIcon fontSize="small" />
                    <Typography variant="caption">
                      Full text available — opens in Zotero reader
                    </Typography>
                  </Stack>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Box
                      sx={{
                        position: "relative",
                        display: "inline-flex",
                        color: "text.disabled",
                      }}
                    >
                      <PictureAsPdfIcon fontSize="small" />
                      <Box
                        sx={{
                          position: "absolute",
                          top: "50%",
                          left: "10%",
                          width: "80%",
                          height: "2px",
                          bgcolor: "currentColor",
                          borderRadius: 1,
                          transform: "translateY(-50%) rotate(-45deg)",
                        }}
                      />
                    </Box>
                    <Typography variant="caption">
                      No full text available
                    </Typography>
                  </Stack>
                </Stack>
              }
              arrow
            >
              <IconButton size="small">
                <StyledLightBulb />
              </IconButton>
            </Tooltip>
          </Stack>
          <Typography variant="body2" color="text.secondary">
            {zoteroConfigured
              ? `Configured for group ${zoteroConfig.group_id}${zoteroConfig.group_slug ? ` (${zoteroConfig.group_slug})` : ""}.`
              : "No credentials configured — Zotero full text lookup is disabled."}
          </Typography>
          {zoteroLoading ? (
            <Skeleton variant="rounded" height={36} width={80} />
          ) : (
            <Button
              size="small"
              onClick={() => setZoteroEditOpen(true)}
              startIcon={<EditIcon />}
            >
              {zoteroConfigured ? "Edit Zotero credentials" : "Add credentials"}
            </Button>
          )}
        </Stack>
      </CardContent>

      <ZoteroEditDialog
        open={zoteroEditOpen}
        onClose={() => setZoteroEditOpen(false)}
        initialConfig={zoteroConfig}
        projectId={project_id}
        onSave={(newConfig) => {
          saveZotero({ project_id, config: newConfig });
          setZoteroEditOpen(false);
        }}
        onDelete={
          zoteroConfigured
            ? () =>
                saveZotero({
                  project_id,
                  config: { api_key: "", group_id: "", group_slug: "" },
                })
            : undefined
        }
      />
    </Card>
  );
};

export default ScreeningCard;
