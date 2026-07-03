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
  Typography,
} from "@mui/material";
import * as React from "react";
import { useContext } from "react";
import { useMutation, useQuery, useQueryClient } from "react-query";

import EditIcon from "@mui/icons-material/Edit";
import { ProjectAPI } from "api";
import { ProjectContext } from "context/ProjectContext";
import { LoadingCardHeader } from "StyledComponents/LoadingCardheader";
import { StyledLightBulb } from "StyledComponents/StyledLightBulb";

const EMPTY_ZOTERO_CONFIG = {
  api_key: "",
  group_id: "",
  group_slug: "",
};

const ZoteroEditDialog = ({ open, onClose, initialConfig, onSave }) => {
  const [state, setState] = React.useState(EMPTY_ZOTERO_CONFIG);

  React.useEffect(() => {
    if (open) {
      setState(
        initialConfig
          ? {
              api_key: initialConfig.api_key || "",
              group_id: initialConfig.group_id || "",
              group_slug: initialConfig.group_slug || "",
            }
          : structuredClone(EMPTY_ZOTERO_CONFIG),
      );
    }
  }, [open, initialConfig]);

  const handleChange = (field) => (e) => {
    setState((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const handleSave = () => {
    onSave({
      api_key: state.api_key.trim(),
      group_id: state.group_id.trim(),
      group_slug: state.group_slug.trim(),
    });
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
            onChange={handleChange("api_key")}
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
          <TextField
            fullWidth
            id="zotero-group-slug"
            label="Group Slug"
            value={state.group_slug}
            onChange={handleChange("group_slug")}
            helperText="URL slug of the group (optional, used for reader links)"
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
          <Typography variant="subtitle1">Zotero full text</Typography>
          <Typography variant="body2" color="text.secondary">
            {zoteroConfigured
              ? `Configured for group ${zoteroConfig.group_id}${zoteroConfig.group_slug ? ` (${zoteroConfig.group_slug})` : ""}.`
              : "No credentials configured — Zotero full text lookup is disabled."}
          </Typography>
          {zoteroLoading ? (
            <Skeleton variant="rounded" height={36} width={80} />
          ) : (
            <Box>
              <Button
                size="small"
                onClick={() => setZoteroEditOpen(true)}
                startIcon={<EditIcon />}
              >
                {zoteroConfigured ? "Edit credentials" : "Add credentials"}
              </Button>
            </Box>
          )}
        </Stack>
      </CardContent>

      <ZoteroEditDialog
        open={zoteroEditOpen}
        onClose={() => setZoteroEditOpen(false)}
        initialConfig={zoteroConfig}
        onSave={(newConfig) => {
          saveZotero({ project_id, config: newConfig });
          setZoteroEditOpen(false);
        }}
      />
    </Card>
  );
};

export default ScreeningCard;
