import React from "react";

import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  Popover,
  Skeleton,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
  FormControlLabel,
} from "@mui/material";
import { ProjectContext } from "context/ProjectContext";
import { useContext } from "react";
import { LoadingCardHeader } from "StyledComponents/LoadingCardheader";

import { ProjectAPI } from "api";
import { useMutation, useQuery, useQueryClient } from "react-query";

import { Add } from "@mui/icons-material";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import BookmarksIcon from "@mui/icons-material/Bookmarks";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import StyleIcon from "@mui/icons-material/Style";
import Grid from "@mui/material/Grid2";
import { StyledLightBulb } from "StyledComponents/StyledLightBulb";
import { TypographySubtitle1Medium } from "StyledComponents/StyledTypography";

import EditIcon from "@mui/icons-material/Edit";
import { useTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";

import { useToggle } from "hooks/useToggle";

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
        <Stack spacing={3}>
          <Box>
            <Typography variant="h6" sx={{ mb: 1 }}>
              Organizing with Tags
            </Typography>
            <Typography variant="body2" align="justify">
              Tags allow you to categorize records based on specific criteria,
              such as reasons for inclusion/exclusion, study characteristics, or
              quality assessment.
            </Typography>
            <Alert severity="info" sx={{ mt: 2 }}>
              Using tags consistently throughout your screening will make your
              data analysis easier after you export your project
            </Alert>
          </Box>

          <Divider />

          <Box>
            <Typography variant="subtitle1" fontWeight="bold" sx={{ mb: 2 }}>
              Tag Structure
            </Typography>
            <Grid container spacing={2}>
              <Grid xs={6}>
                <Box
                  sx={(theme) => ({
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 2,
                    height: "100%",
                    bgcolor:
                      theme.palette.mode === "light"
                        ? "background.paper"
                        : "transparent",
                  })}
                >
                  <Stack spacing={1}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <FolderOpenIcon sx={{ color: "text.secondary" }} />
                      <Typography variant="subtitle2">Tag Groups</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Create categories like "Reasons for Exclusion" or "Study
                      Design"
                    </Typography>
                  </Stack>
                </Box>
              </Grid>
              <Grid xs={6}>
                <Box
                  sx={(theme) => ({
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 2,
                    height: "100%",
                    bgcolor:
                      theme.palette.mode === "light"
                        ? "background.paper"
                        : "transparent",
                  })}
                >
                  <Stack spacing={1}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <BookmarksIcon sx={{ color: "text.secondary" }} />
                      <Typography variant="subtitle2">Tags</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Add specific labels like "Wrong Population" or "Randomized
                      Controlled Trial"
                    </Typography>
                  </Stack>
                </Box>
              </Grid>
              <Grid xs={6}>
                <Box
                  sx={(theme) => ({
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 2,
                    height: "100%",
                    bgcolor:
                      theme.palette.mode === "light"
                        ? "background.paper"
                        : "transparent",
                  })}
                >
                  <Stack spacing={1}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <StyleIcon sx={{ color: "text.secondary" }} />
                      <Typography variant="subtitle2">Organization</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Group related concepts together for better overview
                    </Typography>
                  </Stack>
                </Box>
              </Grid>
            </Grid>
          </Box>

          <Box>
            <Button
              href="https://asreview.readthedocs.io/en/stable/lab/project_create.html#add-tags"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn more
            </Button>
          </Box>
        </Stack>
      </Box>
    </Popover>
  );
};

function labelToExport(label) {
  // Generate a suggested ID based on label
  // since Ids may be used later in data analysis code we suggest simple ascii
  // with no spaces but this is not required
  return label
    .toLowerCase()
    .replaceAll(/\s+/g, "_")
    .replaceAll(/[^a-z0-9_]/g, "");
}

const nowSeconds = () => Date.now() / 1000;

const EMPTY_GROUP = {
  label: "",
  export: "",
  input_helper_text: "",
  single_select: false,
  required_relevant: false,
  required_irrelevant: false,
  require_all: false,
  values: [
    { label: "", export: "", sorted_at: nowSeconds() },
    { label: "", export: "", sorted_at: nowSeconds() },
    { label: "", export: "", sorted_at: nowSeconds() },
  ],
};

/** Normalize a tag group's flags to booleans. */
function normalizeGroup(group) {
  return {
    ...group,
    single_select: Boolean(group.single_select),
    required_relevant: Boolean(group.required_relevant),
    required_irrelevant: Boolean(group.required_irrelevant),
    require_all: Boolean(group.require_all),
  };
}

/**
 * The checklist ("require all options") mode is only meaningful for a
 * multi-select group that is required for at least one decision.
 */
function requireAllAvailable(state) {
  return (
    !state.single_select &&
    (state.required_relevant || state.required_irrelevant)
  );
}

const MutateGroupDialog = ({ project_id, open, onClose, group = null }) => {
  const theme = useTheme();
  const queryClient = useQueryClient();
  const smallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  const [state, setState] = React.useState(
    group ? normalizeGroup(group) : structuredClone(EMPTY_GROUP),
  );

  const { mutate: createTagGroup, error: createError } = useMutation(
    ProjectAPI.createTagGroup,
    {
      mutationKey: ["createTagGroup"],
      onSuccess: () => {
        queryClient.invalidateQueries(["fetchTagGroups", { project_id }]);
        closeDialog();
      },
      onError: (error) => {
        console.error("An error occurred while saving the tag group:", error);
      },
    },
  );

  const { mutate: mutateTagGroup, error: mutateError } = useMutation(
    ProjectAPI.mutateTagGroup,
    {
      mutationKey: ["mutateTagGroup"],
      onSuccess: () => {
        queryClient.invalidateQueries(["fetchTagGroups", { project_id }]);
        closeDialog();
      },
      onError: (error) => {
        console.error("An error occurred while saving the tag group:", error);
      },
    },
  );

  const handleGroupLabelChange = (e) => {
    setState((prev) => ({
      ...prev,
      label: e.target.value,
      export: labelToExport(e.target.value),
    }));
  };

  const handleGroupExportChange = (e) => {
    setState((prev) => ({
      ...prev,
      export: e.target.value,
    }));
  };

  const handleGroupInputHelperTextChange = (e) => {
    setState((prev) => ({
      ...prev,
      input_helper_text: e.target.value,
    }));
  };

  const handleSingleSelectChange = (e) => {
    setState((prev) => {
      const next = { ...prev, single_select: e.target.checked };
      // Checklist mode only applies to multi-select groups.
      if (!requireAllAvailable(next)) {
        next.require_all = false;
      }
      return next;
    });
  };

  const handleRequiredRelevantChange = (e) => {
    setState((prev) => {
      const next = { ...prev, required_relevant: e.target.checked };
      if (!requireAllAvailable(next)) {
        next.require_all = false;
      }
      return next;
    });
  };

  const handleRequiredIrrelevantChange = (e) => {
    setState((prev) => {
      const next = { ...prev, required_irrelevant: e.target.checked };
      if (!requireAllAvailable(next)) {
        next.require_all = false;
      }
      return next;
    });
  };

  const handleRequireAllChange = (e) => {
    setState((prev) => ({
      ...prev,
      require_all: e.target.checked,
    }));
  };

  const handleTagLabelChange = (index, e) => {
    setState((prev) => ({
      ...prev,
      values: prev.values.map((tag, i) =>
        i === index
          ? {
              ...tag,
              label: e.target.value,
              export: labelToExport(e.target.value),
            }
          : tag,
      ),
    }));
  };

  const handleTagExportChange = (index, e) => {
    setState((prev) => ({
      ...prev,
      values: prev.values.map((tag, i) =>
        i === index ? { ...tag, export: e.target.value } : tag,
      ),
    }));
  };

  const handleTagFreeTextChange = (index, e) => {
    setState((prev) => ({
      ...prev,
      values: prev.values.map((tag, i) =>
        i === index
          ? { ...tag, free_text: e.target.checked, free_text_required: false }
          : tag,
      ),
    }));
  };

  const handleTagFreeTextRequiredChange = (index, e) => {
    setState((prev) => ({
      ...prev,
      values: prev.values.map((tag, i) =>
        i === index ? { ...tag, free_text_required: e.target.checked } : tag,
      ),
    }));
  };

  const handleTagMoveToBottom = (index) => {
    setState((prev) => ({
      ...prev,
      values: prev.values.map((tag, i) =>
        i === index ? { ...tag, sorted_at: nowSeconds() } : tag,
      ),
    }));
  };

  const addTag = () => {
    setState((prev) => ({
      ...prev,
      values: [
        ...prev.values,
        {
          label: "",
          export: "",
          sorted_at: nowSeconds(),
        },
      ],
    }));
  };

  const closeDialog = () => {
    if (group == null) {
      setState(structuredClone(EMPTY_GROUP));
    }
    onClose();
  };

  const onSave = () => {
    const payload = {
      ...state,
      values: state.values.filter((tag) => tag.label && tag.export),
    };
    if (group !== null) {
      mutateTagGroup({ project_id, group: payload });
    } else {
      createTagGroup({ project_id, group: payload });
    }
  };

  return (
    <Dialog
      open={open}
      onClose={closeDialog}
      fullScreen={smallScreen}
      fullWidth
      maxWidth="md"
    >
      <DialogTitle>
        {group !== null ? "Edit group of tags" : "Add group of tags"}
      </DialogTitle>
      <DialogContent>
        <Stack spacing={3}>
          <TypographySubtitle1Medium>Group</TypographySubtitle1Medium>
          <Stack direction="row" spacing={3}>
            <TextField
              fullWidth
              id="group-label"
              label="Label"
              value={state.label}
              onChange={handleGroupLabelChange}
              helperText=" "
            />
            <TextField
              fullWidth
              id="group-id"
              label="Export name"
              value={state.export}
              onChange={handleGroupExportChange}
            />
          </Stack>
          <TextField
            fullWidth
            id="group-input-helper-text"
            label="Input helper text"
            value={state.input_helper_text || ""}
            onChange={handleGroupInputHelperTextChange}
            helperText="Optional help text shown below the group header during editing"
          />
          <FormControlLabel
            control={
              <Switch
                checked={Boolean(state.single_select)}
                onChange={handleSingleSelectChange}
              />
            }
            label="Allow only one tag to be selected (radio buttons)"
          />
          <FormControlLabel
            control={
              <Switch
                checked={Boolean(state.required_relevant)}
                onChange={handleRequiredRelevantChange}
              />
            }
            label="Require a selection to mark a record as relevant"
          />
          <FormControlLabel
            control={
              <Switch
                checked={Boolean(state.required_irrelevant)}
                onChange={handleRequiredIrrelevantChange}
              />
            }
            label="Require a selection to mark a record as not relevant"
          />
          <Tooltip
            title={
              requireAllAvailable(state)
                ? "Require every option in this group to be selected (checklist)"
                : "Available for multi-select groups that are required for at least one decision"
            }
          >
            <FormControlLabel
              control={
                <Switch
                  checked={Boolean(state.require_all)}
                  onChange={handleRequireAllChange}
                  disabled={!requireAllAvailable(state)}
                />
              }
              label="Require all options to be selected (checklist)"
            />
          </Tooltip>
        </Stack>
        <Stack spacing={3}>
          <TypographySubtitle1Medium>Tags</TypographySubtitle1Medium>
          {state.values.map((tag, index) => (
            <Stack direction="row" spacing={3} alignItems="center" key={index}>
              <TextField
                fullWidth
                id={`tag-label-${index}`}
                label="Label"
                value={tag.label}
                onChange={(e) => handleTagLabelChange(index, e)}
              />
              <TextField
                fullWidth
                id={`tag-id-${index}`}
                label="Export name"
                value={tag.export}
                onChange={(e) => handleTagExportChange(index, e)}
              />
              <Tooltip title="Allow free text input when this tag is selected">
                <FormControlLabel
                  sx={{ flexShrink: 0, whiteSpace: "nowrap" }}
                  control={
                    <Switch
                      checked={Boolean(tag.free_text)}
                      onChange={(e) => handleTagFreeTextChange(index, e)}
                    />
                  }
                  label="Free text"
                />
              </Tooltip>
              {tag.free_text && (
                <Tooltip title="Require non-empty text when this tag is selected">
                  <FormControlLabel
                    sx={{ flexShrink: 0, whiteSpace: "nowrap" }}
                    control={
                      <Switch
                        checked={Boolean(tag.free_text_required)}
                        onChange={(e) =>
                          handleTagFreeTextRequiredChange(index, e)
                        }
                      />
                    }
                    label="Text required"
                  />
                </Tooltip>
              )}
              <Tooltip title="Move to bottom">
                <IconButton
                  size="small"
                  onClick={() => handleTagMoveToBottom(index)}
                  aria-label="move tag to bottom"
                >
                  <ArrowDownwardIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          ))}
        </Stack>
        <Stack
          direction="row"
          justifyContent="flex-end"
          alignItems="baseline"
          spacing={2}
        >
          <Tooltip title="Add tag">
            <IconButton aria-label="add tag" onClick={addTag}>
              <Add />
            </IconButton>
          </Tooltip>
        </Stack>

        {mutateError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {mutateError?.message}
          </Alert>
        )}
        {createError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {createError?.message}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={closeDialog}>Cancel</Button>
        <Button
          onClick={onSave}
          disabled={
            !state.label ||
            !state.export ||
            state.values.filter((tag) => tag.label && tag.export).length === 0
          }
        >
          {group !== null ? "Save" : "Create Group"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const groupRequirementSummary = (group) => {
  const norm = normalizeGroup(group);
  if (norm.required_relevant && norm.required_irrelevant) {
    return norm.require_all ? "all required" : "required";
  }
  if (norm.required_relevant) {
    return norm.require_all
      ? "all required for relevant"
      : "required for relevant";
  }
  if (norm.required_irrelevant) {
    return norm.require_all
      ? "all required for not relevant"
      : "required for not relevant";
  }
  return null;
};

const Group = ({ project_id, group }) => {
  const [dialogOpen, toggleDialogOpen] = useToggle();

  return (
    <Card sx={{ mb: 2, bgcolor: "background.default" }}>
      <CardHeader
        title={group.label}
        subheader={
          [
            group.single_select ? "Single choice" : null,
            groupRequirementSummary(group),
          ]
            .filter(Boolean)
            .join(" · ") || undefined
        }
        action={
          <Tooltip title="Edit Group">
            <IconButton onClick={toggleDialogOpen}>
              <EditIcon />
            </IconButton>
          </Tooltip>
        }
      />
      <CardContent>
        {group.input_helper_text && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mb: 1, fontStyle: "italic" }}
          >
            {group.input_helper_text}
          </Typography>
        )}
        {group.values.map((t, index) => (
          <Chip
            key={index}
            label={`${t.label} (${t.export})${t.free_text ? " + text" : ""}`}
            sx={{ m: 1 }}
          />
        ))}
      </CardContent>
      <MutateGroupDialog
        key={group.id}
        project_id={project_id}
        open={dialogOpen}
        onClose={toggleDialogOpen}
        group={group}
      />
    </Card>
  );
};

const TagCard = () => {
  const project_id = useContext(ProjectContext);
  const [dialogOpen, toggleDialogOpen] = useToggle();
  const [anchorEl, setAnchorEl] = React.useState(null);

  const { data, isLoading } = useQuery(
    ["fetchTagGroups", { project_id: project_id }],
    ProjectAPI.fetchTagGroups,
    {
      refetchOnWindowFocus: false,
    },
  );

  return (
    <Card>
      <LoadingCardHeader
        title="Labeling tags"
        subheader="Tags and tag groups are used to label records with additional information"
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
        {isLoading ? (
          <Skeleton variant="rectangular" height={56} />
        ) : (
          <>
            {data.length === 0 && (
              <Alert severity="info" sx={{ mb: 2 }}>
                Your tags will appear here
              </Alert>
            )}
            {data.map((c, index) => (
              <Group key={index} group={c} project_id={project_id} />
            ))}
          </>
        )}
      </CardContent>

      <CardContent>
        {isLoading ? (
          <Skeleton variant="rectangular" width={100} height={36} />
        ) : (
          <>
            <MutateGroupDialog
              project_id={project_id}
              open={dialogOpen}
              onClose={toggleDialogOpen}
            />
            <Button onClick={toggleDialogOpen} variant="contained">
              Add tags
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default TagCard;
