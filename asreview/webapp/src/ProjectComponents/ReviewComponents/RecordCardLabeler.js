import {
  Box,
  Button,
  CardActions,
  CardContent,
  Checkbox,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  FormGroup,
  Grid2 as Grid,
  IconButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Paper,
  Radio,
  Stack,
  TextField,
  Tooltip,
  Typography,
  Alert,
} from "@mui/material";
import { alpha } from "@mui/material/styles";
import React from "react";
import { useMutation, useQueryClient } from "react-query";

import { useHotkeys } from "react-hotkeys-hook";

import LibraryAddOutlinedIcon from "@mui/icons-material/LibraryAddOutlined";
import MoreVert from "@mui/icons-material/MoreVert";
import NotInterestedOutlinedIcon from "@mui/icons-material/NotInterestedOutlined";
import NoteAltOutlinedIcon from "@mui/icons-material/NoteAltOutlined";
import FormatColorFillIcon from "@mui/icons-material/FormatColorFill";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import { ProjectAPI } from "api";
import { useToggle } from "hooks/useToggle";
import TimeAgo from "javascript-time-ago";

import { DeleteOutline, LabelOutlined } from "@mui/icons-material";
import en from "javascript-time-ago/locale/en";

TimeAgo.addLocale(en);
const timeAgo = new TimeAgo("en-US");

const formatUser = (user) => {
  if (user?.current_user) {
    return "by you";
  }
  return `by ${user.name}`;
};

const mergeTagValues = (tagsForm, tagValues) => {
  if (!tagsForm) return [];
  if (!tagValues) return structuredClone(tagsForm);
  return tagsForm.map((group) => {
    const savedGroup = tagValues.find((g) => g.id === group.id);
    return {
      ...group,
      values: group.values.map((tag) => {
        const savedTag = savedGroup?.values?.find((t) => t.id === tag.id);
        return {
          ...tag,
          checked: savedTag?.checked || false,
          text: savedTag?.text || "",
        };
      }),
    };
  });
};

/**
 * Whether a group requires a selection for a given decision. Groups can be
 * required for relevant decisions, irrelevant decisions, or both. When ``label``
 * is null/undefined (no decision yet) the group counts as required if it is
 * required for either decision.
 */
const groupRequiredForLabel = (group, label) => {
  const rel = Boolean(group.required_relevant);
  const irr = Boolean(group.required_irrelevant);
  if (label === 1) return rel;
  if (label === 0) return irr;
  return rel || irr;
};

/** Whether a group is configured as a "select all options" checklist. */
const groupRequiresAll = (group) =>
  !group.single_select && Boolean(group.require_all);

/**
 * Whether the requirement of a single group is satisfied for the given decision.
 */
const groupRequirementMet = (group, label) => {
  if (!groupRequiredForLabel(group, label)) return true;
  const values = group.values || [];
  if (groupRequiresAll(group)) {
    return (
      values.length > 0 &&
      values.every(
        (t) =>
          t.checked &&
          (!t.free_text_required || (t.text && t.text.trim() !== "")),
      )
    );
  }
  return values.some(
    (t) =>
      t.checked && (!t.free_text_required || (t.text && t.text.trim() !== "")),
  );
};

/**
 * Returns true when every group's requirement is met for the given decision.
 * Used to gate labeling/saving when a selection is obligatory.
 */
const tagRequirementsMetForLabel = (tagValues, label) =>
  (tagValues || []).every((group) => groupRequirementMet(group, label));

/**
 * Explains the "*" marker shown next to required tag groups. Renders nothing
 * when no group in the form is required for any decision.
 */
const TagRequiredLegend = ({ tagsForm }) => {
  if (
    !Array.isArray(tagsForm) ||
    !tagsForm.some((g) => groupRequiredForLabel(g, null))
  ) {
    return null;
  }
  return (
    <Typography
      variant="caption"
      color="text.secondary"
      sx={{ display: "block", mt: 1 }}
    >
      * a selection must be made in this group
    </Typography>
  );
};

/**
 * Renders the input controls for a single tag group.
 *
 * In `readOnly` mode the group is shown compactly: only the selected values are
 * rendered as chips, and a placeholder is shown when nothing is selected.
 *
 * Otherwise, regular groups are rendered as checkboxes (multiple selectable) and
 * groups with `single_select` as radio buttons (only one selectable). When a
 * single-select group is not `required`, clicking the selected radio button
 * deselects it. If a single-select group already has more than one value
 * selected (invalid data, e.g. created by an older version or another client),
 * all selected values are still shown and a warning is displayed.
 */
const TagGroupInput = ({
  group,
  groupValues,
  onToggle,
  onSelectExclusive,
  onTextChange,
  disabled = false,
  readOnly = false,
  label = null,
}) => {
  const totalCount = (groupValues?.values || []).length;
  const checkedCount = (groupValues?.values || []).filter(
    (t) => t.checked,
  ).length;
  const singleSelect = Boolean(group.single_select);
  const requireAll = groupRequiresAll(group);
  const requiredAny = groupRequiredForLabel(group, null);
  // In read-only mode the decision is known, so only enforce its requirement;
  // while deciding (label null) flag what is required for either decision.
  const requiredForThis = groupRequiredForLabel(group, label);
  const allChecked = totalCount > 0 && checkedCount === totalCount;
  const invalid = singleSelect && checkedCount > 1;
  const missing =
    requiredForThis && (requireAll ? !allChecked : checkedCount === 0);
  // Explains which decision(s) the selection is required for.
  const requirementReason = [
    group.required_relevant && "for allowing relevant decision",
    group.required_irrelevant && "for allowing irrelevant decision",
  ]
    .filter(Boolean)
    .join(" and ");

  if (readOnly) {
    const selected = group.values
      .map((tag, j) => ({ tag, value: groupValues?.values[j] }))
      .filter(({ value }) => value?.checked);
    return (
      <Stack direction="column" spacing={1}>
        <Typography variant="h6">
          {group.label}
          {requiredAny && " *"}
        </Typography>
        {invalid && (
          <Alert severity="warning">
            More than one option is selected in this single-choice group.
          </Alert>
        )}
        {missing && (
          <Alert severity="warning">
            {requireAll
              ? "All options must be selected in this group, but some are missing."
              : "A selection is required in this group, but none is made."}
          </Alert>
        )}
        {selected.length > 0 ? (
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            {selected.map(({ tag, value }) => (
              <Chip
                key={`${group.id}:${tag.id}`}
                label={value?.text ? `${tag.label}: ${value.text}` : tag.label}
              />
            ))}
          </Stack>
        ) : (
          <Typography variant="body2" color="text.secondary">
            None selected
          </Typography>
        )}
      </Stack>
    );
  }

  return (
    <Stack direction="column" spacing={1}>
      <Typography variant="h6">
        {group.label}
        {requiredAny && " *"}
      </Typography>
      {group.input_helper_text && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontStyle: "italic" }}
        >
          {group.input_helper_text}
        </Typography>
      )}
      {invalid && (
        <Alert severity="warning">
          More than one option is selected in this single-choice group.
        </Alert>
      )}
      {missing && (
        <Typography variant="caption" color="warning">
          {(requireAll
            ? "Select all options"
            : singleSelect
              ? "Select one option"
              : "Select at least one option") +
            (requirementReason ? ` ${requirementReason}` : "")}
        </Typography>
      )}
      <FormGroup row={false}>
        {group.values.map((tag, j) => {
          const checked = groupValues?.values[j]?.checked || false;
          const text = groupValues?.values[j]?.text || "";
          return (
            <Box key={`${group.id}:${tag.id}`}>
              <FormControlLabel
                control={
                  singleSelect ? (
                    <Radio
                      checked={checked}
                      onChange={() => onSelectExclusive(group.id, tag.id)}
                      onClick={() => {
                        // A radio button can always be deselected by clicking
                        // the already-selected option again.
                        if (checked) {
                          onToggle(false, group.id, tag.id);
                        }
                      }}
                      disabled={disabled}
                    />
                  ) : (
                    <Checkbox
                      checked={checked}
                      onChange={(e) =>
                        onToggle(e.target.checked, group.id, tag.id)
                      }
                      disabled={disabled}
                    />
                  )
                }
                label={tag.label}
              />
              {tag.free_text && checked && (
                <TextField
                  size="small"
                  fullWidth
                  variant="standard"
                  placeholder="Add free text…"
                  value={text}
                  onChange={(e) =>
                    onTextChange?.(group.id, tag.id, e.target.value)
                  }
                  disabled={disabled}
                  error={
                    tag.free_text_required && (!text || text.trim() === "")
                  }
                  helperText={
                    tag.free_text_required && (!text || text.trim() === "")
                      ? "Text is required for this selection"
                      : undefined
                  }
                  sx={{ ml: 4, mb: 1, maxWidth: "calc(100% - 32px)" }}
                />
              )}
            </Box>
          );
        })}
      </FormGroup>
    </Stack>
  );
};

/** Remove characters that are not allowed in a list item name. */
const sanitizeListItemName = (name) => (name || "").replace(/[,;]/g, "");

/** Generate a uuid_v4 for a new list item. */
const newItemId = () => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID.
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

/** Current unix timestamp (in seconds) used to order list items. */
const nowSeconds = () => Date.now() / 1000;

/** Whether a list item has no (non-whitespace) text. */
const isEmptyItem = (item) => !item.name || !item.name.trim();

/**
 * Create a fresh empty input row for a list. It has no ``sorted_at`` timestamp
 * yet: a timestamp is only assigned once the user first edits it, which keeps
 * the untouched input row pinned to the bottom of the list.
 */
const newEmptyListItem = (listId) => ({
  list_id: listId,
  item_id: newItemId(),
  name: "",
  sorted_at: null,
});

/** Items belonging to a given list. */
const itemsForList = (listValues, listId) =>
  (listValues || []).filter((item) => item.list_id === listId);

/**
 * Items of a list ordered for display: by their ``sorted_at`` timestamp (oldest
 * first). Items without a timestamp (the untouched input row) always sort to
 * the very bottom. Empty rows that already have a timestamp keep their place
 * and are only removed by an explicit delete or when saving.
 */
const sortedListItems = (items) =>
  [...items].sort((a, b) => {
    if (a.sorted_at == null && b.sorted_at == null) return 0;
    if (a.sorted_at == null) return 1;
    if (b.sorted_at == null) return -1;
    return a.sorted_at - b.sorted_at;
  });

/**
 * Ensure every configured list ends with an empty input row: add a fresh
 * (timestamp-less) row when the list has no items yet or its last row already
 * contains text. There is no separate "add item" button.
 */
const withTrailingEmptyItems = (listValues, listsForm) => {
  const result = [...(listValues || [])];
  (listsForm || []).forEach((list) => {
    const ordered = sortedListItems(
      result.filter((item) => item.list_id === list.id),
    );
    const last = ordered[ordered.length - 1];
    if (!last || !isEmptyItem(last)) {
      result.push(newEmptyListItem(list.id));
    }
  });
  return result;
};

/**
 * Apply a name change to one list item. A timestamp is assigned the first time
 * a previously untouched (timestamp-less) row receives non-empty text, which
 * fixes its position in the list. Already timestamped rows keep their order even
 * when cleared, so they only move on an explicit reset or are dropped on save.
 */
const applyListItemName = (items, itemId, name) =>
  items.map((item) => {
    if (item.item_id !== itemId) return item;
    const next = { ...item, name };
    if (next.sorted_at == null && !isEmptyItem(next)) {
      next.sorted_at = nowSeconds();
    }
    return next;
  });

/** Drop empty rows and trim names; used right before saving. */
const cleanListValues = (listValues) =>
  (listValues || [])
    .filter((item) => !isEmptyItem(item))
    .map((item) => ({
      ...item,
      name: item.name.trim(),
      sorted_at: item.sorted_at ?? nowSeconds(),
    }));

/** Whether a list is required for the given decision (relevant only). */
const listRequiredForLabel = (list, label) => {
  const rel = Boolean(list.required_for_relevant);
  if (label === 0) return false;
  // Required for relevant decisions, and flagged while still deciding.
  return rel;
};

/** Whether a single list's requirement is met for the given decision. */
const listRequirementMet = (list, items, label) => {
  if (!listRequiredForLabel(list, label)) return true;
  return items.some((item) => !isEmptyItem(item));
};

/** Whether every list's requirement is met for the given decision. */
const listRequirementsMetForLabel = (listsForm, listValues, label) =>
  (listsForm || []).every((list) =>
    listRequirementMet(list, itemsForList(listValues, list.id), label),
  );

/**
 * Explains the "*" marker shown next to required lists. Renders nothing when no
 * list is required.
 */
const ListRequiredLegend = ({ listsForm }) => {
  if (
    !Array.isArray(listsForm) ||
    !listsForm.some((list) => listRequiredForLabel(list, null))
  ) {
    return null;
  }
  return (
    <Typography
      variant="caption"
      color="text.secondary"
      sx={{ display: "block", mt: 1 }}
    >
      * at least one item must be added for allowing relevant decision
    </Typography>
  );
};

/**
 * Renders the input controls for a single list. In `readOnly` mode the list is
 * shown compactly: only its items are rendered as chips, with a placeholder when
 * empty. Otherwise each item is an editable text field that can be removed, with
 * an always-present empty input row at the bottom and a button to reset an
 * item's timestamp so it moves to the bottom of the list.
 */
const ListGroupInput = ({
  list,
  items,
  onChangeItem,
  onRemoveItem,
  onResetItem,
  disabled = false,
  readOnly = false,
  label = null,
}) => {
  const required = Boolean(list.required_for_relevant);
  const ordered = sortedListItems(items);
  const inputRefs = React.useRef({});

  // Pressing Enter in a text field jumps to the first empty row so items can be
  // added in quick succession without reaching for the mouse.
  const focusFirstEmpty = () => {
    requestAnimationFrame(() => {
      const target = sortedListItems(items).find((item) => isEmptyItem(item));
      const input = target && inputRefs.current[target.item_id];
      if (input) input.focus();
    });
  };

  if (readOnly) {
    const filled = ordered.filter((item) => !isEmptyItem(item));
    return (
      <Stack direction="column" spacing={1}>
        <Typography variant="h6">
          {list.name}
          {required && " *"}
        </Typography>
        {filled.length > 0 ? (
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            {filled.map((item) => (
              <Chip key={item.item_id} label={item.name} />
            ))}
          </Stack>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No items
          </Typography>
        )}
      </Stack>
    );
  }

  const missing =
    listRequiredForLabel(list, label) &&
    !items.some((item) => !isEmptyItem(item));

  return (
    <Stack direction="column" spacing={1}>
      <Typography variant="h6">
        {list.name}
        {required && " *"}
      </Typography>
      {list.input_helper_text && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontStyle: "italic" }}
        >
          {list.input_helper_text}
        </Typography>
      )}
      {missing && (
        <Typography variant="caption" color="warning">
          Add at least one item for allowing relevant decision
        </Typography>
      )}
      {ordered.map((item) => {
        const empty = isEmptyItem(item);
        return (
          <Stack
            direction="row"
            spacing={1}
            alignItems="center"
            key={item.item_id}
          >
            <TextField
              size="small"
              fullWidth
              variant="standard"
              placeholder="Add item…"
              value={item.name}
              inputRef={(el) => {
                if (el) inputRefs.current[item.item_id] = el;
                else delete inputRefs.current[item.item_id];
              }}
              onChange={(e) =>
                onChangeItem(item.item_id, sanitizeListItemName(e.target.value))
              }
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  focusFirstEmpty();
                }
              }}
              disabled={disabled}
            />
            {!empty && (
              <>
                <Tooltip title="Move to bottom (reset timestamp to now)">
                  <span>
                    <IconButton
                      size="small"
                      onClick={() => onResetItem(item.item_id)}
                      disabled={disabled}
                      aria-label="move item to bottom"
                    >
                      <ArrowDownwardIcon fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
                <Tooltip title="Remove item">
                  <span>
                    <IconButton
                      size="small"
                      onClick={() => onRemoveItem(item.item_id)}
                      disabled={disabled}
                      aria-label="remove item"
                    >
                      <DeleteOutline fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
              </>
            )}
          </Stack>
        );
      })}
    </Stack>
  );
};

const ListsDialog = ({
  project_id,
  record_id,
  label,
  listsForm,
  listValues,
  tagValues,
  retrainAfterDecision,
  open,
  onClose,
  onSave,
}) => {
  const queryClient = useQueryClient();
  const [localListValues, setLocalListValues] = React.useState(
    withTrailingEmptyItems(structuredClone(listValues || []), listsForm),
  );

  React.useEffect(() => {
    if (open) {
      setLocalListValues(
        withTrailingEmptyItems(structuredClone(listValues || []), listsForm),
      );
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleChangeItem = (itemId, name) => {
    setLocalListValues((prev) =>
      withTrailingEmptyItems(applyListItemName(prev, itemId, name), listsForm),
    );
  };

  const handleRemoveItem = (itemId) => {
    setLocalListValues((prev) =>
      withTrailingEmptyItems(
        prev.filter((item) => item.item_id !== itemId),
        listsForm,
      ),
    );
  };

  const handleResetItem = (itemId) => {
    setLocalListValues((prev) =>
      prev.map((item) =>
        item.item_id === itemId ? { ...item, sorted_at: nowSeconds() } : item,
      ),
    );
  };

  const { isError, isLoading, mutate } = useMutation(
    ProjectAPI.mutateClassification,
    {
      onSuccess: () => {
        queryClient.invalidateQueries(["fetchLabeledRecord", { project_id }]);
        onSave(localListValues);
        onClose();
      },
    },
  );

  const decisionLabel = label === 1 || label === 0 ? label : null;
  const requirementsMet = listRequirementsMetForLabel(
    listsForm,
    localListValues,
    decisionLabel,
  );

  return (
    <Dialog open={open} onClose={onClose} fullWidth>
      <DialogTitle>Edit lists</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ pt: 1 }}>
          {listsForm &&
            listsForm.map((list) => (
              <ListGroupInput
                key={list.id}
                list={list}
                items={itemsForList(localListValues, list.id)}
                onChangeItem={handleChangeItem}
                onRemoveItem={handleRemoveItem}
                onResetItem={handleResetItem}
                disabled={isLoading}
                label={decisionLabel}
              />
            ))}
        </Stack>
        <ListRequiredLegend listsForm={listsForm} />
        {isError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Failed to update lists.
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button
          onClick={() =>
            mutate({
              project_id,
              record_id,
              label,
              tagValues,
              listValues: cleanListValues(localListValues),
              retrain_model: retrainAfterDecision,
              post: false,
            })
          }
          color="primary"
          disabled={isLoading || !requirementsMet}
        >
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const NoteDialog = ({ project_id, record_id, open, onClose, note = null }) => {
  const queryClient = useQueryClient();

  const [noteState, setNoteState] = React.useState(note);

  React.useEffect(() => {
    if (open) {
      setNoteState(note);
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const { isError, isLoading, mutate } = useMutation(ProjectAPI.mutateNote, {
    onSuccess: () => {
      queryClient.invalidateQueries(["fetchLabeledRecord", { project_id }]);
      queryClient.setQueryData(["fetchRecord", { project_id }], (data) => {
        return {
          ...data,
          result: {
            ...data.result,
            state: {
              ...data.result.state,
              note: noteState,
            },
          },
        };
      });
      onClose();
    },
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      disableRestoreFocus // bug https://github.com/mui/material-ui/issues/33004
    >
      <DialogTitle>Add note</DialogTitle>
      <DialogContent>
        <TextField
          autoComplete="off"
          id="record-note"
          autoFocus
          fullWidth
          multiline
          onChange={(event) => setNoteState(event.target.value)}
          onFocus={(e) =>
            e.currentTarget.setSelectionRange(
              e.currentTarget.value.length,
              e.currentTarget.value.length,
            )
          } // bug https://github.com/mui/material-ui/issues/12779
          placeholder="Write a note for this record..."
          rows={4}
          value={noteState ? noteState : ""}
          error={isError}
          disabled={isLoading}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button
          onClick={() => {
            mutate({
              project_id: project_id,
              record_id: record_id,
              note: noteState,
            });
          }}
          color="primary"
          disabled={isLoading || noteState === note}
        >
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const TagsDialog = ({
  project_id,
  record_id,
  label,
  tagsForm,
  tagValues,
  retrainAfterDecision,
  open,
  onClose,
  onSave,
}) => {
  const queryClient = useQueryClient();
  const [localTagValues, setLocalTagValues] = React.useState(
    mergeTagValues(tagsForm, tagValues),
  );

  React.useEffect(() => {
    if (open) {
      setLocalTagValues(mergeTagValues(tagsForm, tagValues));
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTagValueChange = (isChecked, groupId, tagId) => {
    let groupI = localTagValues.findIndex((group) => group.id === groupId);
    if (groupI === -1) return;
    let tagI = localTagValues[groupI].values.findIndex(
      (tag) => tag.id === tagId,
    );
    if (tagI === -1) return;
    let copy = structuredClone(localTagValues);
    copy[groupI].values[tagI]["checked"] = isChecked;
    setLocalTagValues(copy);
  };

  const handleSingleSelect = (groupId, tagId) => {
    let groupI = localTagValues.findIndex((group) => group.id === groupId);
    if (groupI === -1) return;
    let copy = structuredClone(localTagValues);
    copy[groupI].values = copy[groupI].values.map((tag) => ({
      ...tag,
      checked: tag.id === tagId,
    }));
    setLocalTagValues(copy);
  };

  const handleTagTextChange = (groupId, tagId, text) => {
    let groupI = localTagValues.findIndex((group) => group.id === groupId);
    if (groupI === -1) return;
    let tagI = localTagValues[groupI].values.findIndex(
      (tag) => tag.id === tagId,
    );
    if (tagI === -1) return;
    let copy = structuredClone(localTagValues);
    copy[groupI].values[tagI]["text"] = text;
    setLocalTagValues(copy);
  };

  const { isError, isLoading, mutate } = useMutation(
    ProjectAPI.mutateClassification,
    {
      onSuccess: () => {
        queryClient.invalidateQueries(["fetchLabeledRecord", { project_id }]);
        onSave(localTagValues);
        onClose();
      },
    },
  );

  return (
    <Dialog open={open} onClose={onClose} fullWidth>
      <DialogTitle>Edit tags</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} columns={2} sx={{ pt: 1 }}>
          {tagsForm &&
            tagsForm.map((group, i) => (
              <Grid size={2} key={group.id}>
                <TagGroupInput
                  group={group}
                  groupValues={localTagValues[i]}
                  onToggle={handleTagValueChange}
                  onSelectExclusive={handleSingleSelect}
                  onTextChange={handleTagTextChange}
                  disabled={isLoading}
                  label={label === 1 || label === 0 ? label : null}
                />
              </Grid>
            ))}
        </Grid>
        <TagRequiredLegend tagsForm={tagsForm} />
        {isError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Failed to update tags.
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button
          onClick={() =>
            mutate({
              project_id,
              record_id,
              label,
              tagValues: localTagValues,
              retrain_model: retrainAfterDecision,
              post: false,
            })
          }
          color="primary"
          disabled={
            isLoading ||
            !tagRequirementsMetForLabel(
              localTagValues,
              label === 1 || label === 0 ? label : null,
            )
          }
        >
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const RecordCardLabeler = ({
  project_id,
  record_id,
  label,
  labelFromDataset = null,
  tagsForm,
  tagValues = null,
  listsForm = null,
  listValues = null,
  note = null,
  showNotes = true,
  labelTime = null,
  user = null,
  onDecisionClose = null,
  hotkeys = false,
  landscape = false,
  retrainAfterDecision = true,
  changeDecision = true,
  highlightAvailable = false,
  highlightOn = false,
  onToggleHighlight = null,
}) => {
  const [editState] = useToggle(!(label === 1 || label === 0));
  const [showNotesDialog, toggleShowNotesDialog] = useToggle(false);
  const [showTagsDialog, toggleShowTagsDialog] = useToggle(false);
  const [showListsDialog, toggleShowListsDialog] = useToggle(false);
  const [tagValuesState, setTagValuesState] = React.useState(
    mergeTagValues(tagsForm, tagValues),
  );
  const [listValuesState, setListValuesState] = React.useState(
    withTrailingEmptyItems(structuredClone(listValues || []), listsForm),
  );

  // `listsForm` can arrive after this component first mounts (e.g. the record
  // query resolves before the form config). The one-time `useState` initializer
  // above would then run with an empty form and add no trailing input row, so
  // re-apply it once the form becomes available. `withTrailingEmptyItems` only
  // appends a row when needed, so this never clobbers user edits.
  React.useEffect(() => {
    setListValuesState((prev) => withTrailingEmptyItems(prev, listsForm));
  }, [listsForm]);

  const { error, isError, isLoading, mutate, isSuccess } = useMutation(
    ProjectAPI.mutateClassification,
    {
      onSuccess: () => {
        if (onDecisionClose) {
          onDecisionClose();
        }
      },
    },
  );

  const handleTagValueChange = (isChecked, groupId, tagId) => {
    let groupI = tagValuesState.findIndex((group) => group.id === groupId);
    if (groupI === -1) return;
    let tagI = tagValuesState[groupI].values.findIndex(
      (tag) => tag.id === tagId,
    );
    if (tagI === -1) return;

    let tagValuesCopy = structuredClone(tagValuesState);
    tagValuesCopy[groupI].values[tagI]["checked"] = isChecked;

    setTagValuesState(tagValuesCopy);
  };

  const handleSingleSelect = (groupId, tagId) => {
    let groupI = tagValuesState.findIndex((group) => group.id === groupId);
    if (groupI === -1) return;

    let tagValuesCopy = structuredClone(tagValuesState);
    tagValuesCopy[groupI].values = tagValuesCopy[groupI].values.map((tag) => ({
      ...tag,
      checked: tag.id === tagId,
    }));

    setTagValuesState(tagValuesCopy);
  };

  const handleTagTextChange = (groupId, tagId, text) => {
    let groupI = tagValuesState.findIndex((group) => group.id === groupId);
    if (groupI === -1) return;
    let tagI = tagValuesState[groupI].values.findIndex(
      (tag) => tag.id === tagId,
    );
    if (tagI === -1) return;

    let tagValuesCopy = structuredClone(tagValuesState);
    tagValuesCopy[groupI].values[tagI]["text"] = text;

    setTagValuesState(tagValuesCopy);
  };

  const handleChangeListItem = (itemId, name) => {
    setListValuesState((prev) =>
      withTrailingEmptyItems(applyListItemName(prev, itemId, name), listsForm),
    );
  };

  const handleRemoveListItem = (itemId) => {
    setListValuesState((prev) =>
      withTrailingEmptyItems(
        prev.filter((item) => item.item_id !== itemId),
        listsForm,
      ),
    );
  };

  const handleResetListItem = (itemId) => {
    setListValuesState((prev) =>
      prev.map((item) =>
        item.item_id === itemId ? { ...item, sorted_at: nowSeconds() } : item,
      ),
    );
  };

  // Requirements can differ per decision, so check relevant and irrelevant
  // independently to gate each button.
  const relevantRequirementsMet =
    tagRequirementsMetForLabel(tagValuesState, 1) &&
    listRequirementsMetForLabel(listsForm, listValuesState, 1);
  const irrelevantRequirementsMet =
    tagRequirementsMetForLabel(tagValuesState, 0) &&
    listRequirementsMetForLabel(listsForm, listValuesState, 0);

  const makeDecision = (label) => {
    if (!tagRequirementsMetForLabel(tagValuesState, label)) return;
    if (!listRequirementsMetForLabel(listsForm, listValuesState, label)) return;
    mutate({
      project_id: project_id,
      record_id: record_id,
      label: label,
      tagValues: tagValuesState,
      listValues: cleanListValues(listValuesState),
      retrain_model: retrainAfterDecision,
      post: editState,
    });
  };

  const [anchorEl, setAnchorEl] = React.useState(null);
  const openMenu = Boolean(anchorEl);

  useHotkeys("r", () => hotkeys && !isLoading && !isSuccess && makeDecision(1));
  useHotkeys("i", () => hotkeys && !isLoading && !isSuccess && makeDecision(0));
  useHotkeys(
    "n",
    () => hotkeys && !isLoading && !isSuccess && toggleShowNotesDialog(),
    { keyup: true },
  );

  return (
    <Stack
      sx={(theme) => ({
        bgcolor: alpha(
          label === 1
            ? alpha(theme.palette.tertiary.main, 1)
            : label === 0
              ? alpha(theme.palette.grey[600], 1)
              : alpha(theme.palette.secondary.dark, 1),

          theme.palette.action.selectedOpacity * 1.5,
        ),
        justifyContent: "space-between",
        alignItems: "stretch",
        height: "100%",
      })}
    >
      <Box>
        {Array.isArray(tagsForm) && tagsForm.length > 0 && (
          <CardContent>
            <Grid container spacing={2} columns={2}>
              {tagsForm &&
                tagsForm.map((group, i) => (
                  <Grid
                    size={
                      tagsForm.length === 1
                        ? 2
                        : landscape
                          ? 2
                          : { xs: 2, sm: 1 }
                    }
                    key={group.id}
                  >
                    <TagGroupInput
                      group={group}
                      groupValues={tagValuesState[i]}
                      onToggle={handleTagValueChange}
                      onSelectExclusive={handleSingleSelect}
                      onTextChange={handleTagTextChange}
                      readOnly={!editState}
                      label={label === 1 || label === 0 ? label : null}
                      disabled={
                        !editState || !changeDecision || isLoading || isSuccess
                      }
                    />
                  </Grid>
                ))}
            </Grid>
            <TagRequiredLegend tagsForm={tagsForm} />
          </CardContent>
        )}
      </Box>
      {Array.isArray(listsForm) && listsForm.length > 0 && (
        <Box>
          <Divider />
          <CardContent>
            <Stack spacing={3}>
              {listsForm.map((list) => (
                <ListGroupInput
                  key={list.id}
                  list={list}
                  items={itemsForList(listValuesState, list.id)}
                  onChangeItem={handleChangeListItem}
                  onRemoveItem={handleRemoveListItem}
                  onResetItem={handleResetListItem}
                  readOnly={!editState}
                  label={label === 1 || label === 0 ? label : null}
                  disabled={
                    !editState || !changeDecision || isLoading || isSuccess
                  }
                />
              ))}
            </Stack>
            {editState && <ListRequiredLegend listsForm={listsForm} />}
          </CardContent>
        </Box>
      )}
      <Box>
        {(note !== null || labelFromDataset !== null) && (
          <>
            <Divider />
            <CardContent>
              {note && (
                <Paper
                  elevation={0}
                  sx={{
                    p: 2,
                    mb: 2,
                    bgcolor: "background.default",
                  }}
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <NoteAltOutlinedIcon />
                    <Typography variant="subtitle1">Note</Typography>
                  </Stack>
                  <Typography sx={{ mt: 1, whiteSpace: "pre-wrap" }}>
                    {note}
                  </Typography>
                </Paper>
              )}
              {labelFromDataset === 0 && (
                <Paper
                  elevation={0}
                  sx={{
                    p: 2,
                    bgcolor: "background.default",
                  }}
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <LabelOutlined />
                    <Typography variant="subtitle1">Not relevant</Typography>
                  </Stack>
                  <Typography sx={{ mt: 1 }}>
                    This record is labeled as not relevant in the dataset
                  </Typography>
                </Paper>
              )}
              {labelFromDataset === 1 && (
                <Paper
                  elevation={0}
                  sx={{
                    p: 2,
                    bgcolor: "background.default",
                  }}
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <LabelOutlined />
                    <Typography variant="subtitle1">Relevant</Typography>
                  </Stack>
                  <Typography sx={{ mt: 1 }}>
                    This record is labeled as relevant in the dataset
                  </Typography>
                </Paper>
              )}
            </CardContent>
          </>
        )}

        {isError && (
          <CardContent>
            <Alert severity="error">
              Failed to label record. {error?.message}
            </Alert>
          </CardContent>
        )}
        <CardActions
          sx={(theme) => ({
            bgcolor:
              label === 1
                ? alpha(theme.palette.tertiary.main, 1)
                : label === 0
                  ? alpha(theme.palette.grey[600], 1)
                  : null,
          })}
        >
          {editState && (
            <>
              <Tooltip
                title="Label as relevant (keyboard shortcut: R)"
                enterDelay={2000}
                leaveDelay={200}
                placement="bottom"
              >
                <Button
                  id="relevant"
                  onClick={() => makeDecision(1)}
                  variant="contained"
                  startIcon={<LibraryAddOutlinedIcon />}
                  disabled={isLoading || isSuccess || !relevantRequirementsMet}
                  sx={(theme) => ({
                    color: theme.palette.getContrastText(
                      theme.palette.tertiary.main,
                    ),
                    bgcolor: theme.palette.tertiary.main,
                  })}
                >
                  Relevant
                </Button>
              </Tooltip>
              <Tooltip
                title="Label as irrelevant (keyboard shortcut: I)"
                enterDelay={2000}
                leaveDelay={200}
                placement="bottom"
              >
                <Button
                  id="irrelevant"
                  onClick={() => makeDecision(0)}
                  startIcon={<NotInterestedOutlinedIcon />}
                  disabled={
                    isLoading || isSuccess || !irrelevantRequirementsMet
                  }
                  variant="contained"
                  color="grey.600"
                >
                  Not relevant
                </Button>
              </Tooltip>
            </>
          )}
          {(label === 1 || label === 0) && (
            <>
              {!landscape && (
                <Typography
                  variant="secondary"
                  sx={(theme) => ({
                    pl: 1,
                    color:
                      label === 1
                        ? theme.palette.getContrastText(
                            theme.palette.tertiary.main,
                          )
                        : label === 0
                          ? theme.palette.getContrastText(
                              theme.palette.grey[600],
                            )
                          : theme.palette.text.primary,
                  })}
                >
                  Labeled {label === 1 ? "relevant" : "not relevant"}{" "}
                  {user && formatUser(user)}{" "}
                  {labelTime != null
                    ? timeAgo.format(new Date(labelTime * 1000))
                    : "some time ago"}
                </Typography>
              )}
            </>
          )}
          <Box sx={{ flexGrow: 1 }} />

          {highlightAvailable && (
            <Tooltip
              title={
                highlightOn
                  ? "Hide keyword highlights"
                  : "Show keyword highlights"
              }
              placement="bottom"
            >
              <IconButton
                onClick={onToggleHighlight}
                aria-label="toggle keyword highlighting"
                sx={{ color: highlightOn ? "primary.main" : undefined }}
              >
                <FormatColorFillIcon />
              </IconButton>
            </Tooltip>
          )}

          {editState && showNotes && (
            <>
              <Tooltip
                title="Add note (keyboard shortcut: N)"
                enterDelay={2000}
                leaveDelay={200}
                placement="bottom"
              >
                <IconButton
                  onClick={toggleShowNotesDialog}
                  aria-label="add note"
                  disabled={isLoading || isSuccess}
                  sx={(theme) => ({
                    // color: theme.palette.getContrastText(
                    //   theme.palette.secondary.dark,
                    // ),
                  })}
                >
                  <NoteAltOutlinedIcon />
                </IconButton>
              </Tooltip>
            </>
          )}

          {(label === 1 || label === 0) && changeDecision && (
            <>
              <Tooltip title="Options">
                <IconButton
                  id="card-positioned-button"
                  aria-controls={openMenu ? "card-positioned-menu" : undefined}
                  aria-haspopup="true"
                  aria-expanded={openMenu ? "true" : undefined}
                  onClick={(event) => setAnchorEl(event.currentTarget)}
                  sx={(theme) => ({
                    color:
                      label === 1
                        ? theme.palette.getContrastText(
                            theme.palette.tertiary.main,
                          )
                        : label === 0
                          ? theme.palette.getContrastText(
                              theme.palette.grey[600],
                            )
                          : theme.palette.action.primary,
                  })}
                >
                  <MoreVert />
                </IconButton>
              </Tooltip>

              <Menu
                id="card-positioned-menu"
                aria-labelledby="card-positioned-button"
                anchorEl={anchorEl}
                open={openMenu}
                onClose={() => setAnchorEl(null)}
                anchorOrigin={{
                  vertical: "bottom",
                  horizontal: "right",
                }}
                transformOrigin={{
                  vertical: "bottom",
                  horizontal: "right",
                }}
              >
                {/* toggle label */}
                {(label === 1 || label === 0) && (
                  <MenuItem onClick={() => makeDecision(label === 1 ? 0 : 1)}>
                    <ListItemIcon>
                      {label === 1 ? (
                        <NotInterestedOutlinedIcon />
                      ) : (
                        <LibraryAddOutlinedIcon />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        label === 1
                          ? "Change to Not Relevant"
                          : "Change to Relevant"
                      }
                    />
                  </MenuItem>
                )}
                {Array.isArray(tagsForm) && tagsForm.length > 0 && (
                  <MenuItem
                    onClick={() => {
                      toggleShowTagsDialog();
                      setAnchorEl(null);
                    }}
                  >
                    <ListItemIcon>
                      <LabelOutlined />
                    </ListItemIcon>
                    <ListItemText primary="Edit tags" />
                  </MenuItem>
                )}
                {Array.isArray(listsForm) && listsForm.length > 0 && (
                  <MenuItem
                    onClick={() => {
                      toggleShowListsDialog();
                      setAnchorEl(null);
                    }}
                  >
                    <ListItemIcon>
                      <LabelOutlined />
                    </ListItemIcon>
                    <ListItemText primary="Edit lists" />
                  </MenuItem>
                )}
                <MenuItem
                  onClick={() => {
                    toggleShowNotesDialog();
                    setAnchorEl(null);
                  }}
                >
                  <ListItemIcon>
                    <NoteAltOutlinedIcon />
                  </ListItemIcon>
                  <ListItemText primary={note ? "Change note" : "Add note"} />
                </MenuItem>
                <MenuItem onClick={() => {}} disabled>
                  <ListItemIcon>
                    <DeleteOutline />
                  </ListItemIcon>
                  <ListItemText
                    primary={"Remove my label"}
                    secondary={"Coming soon"}
                  />
                </MenuItem>
              </Menu>
            </>
          )}
          <NoteDialog
            project_id={project_id}
            record_id={record_id}
            open={showNotesDialog}
            onClose={toggleShowNotesDialog}
            note={note}
          />
          {Array.isArray(tagsForm) && tagsForm.length > 0 && (
            <TagsDialog
              project_id={project_id}
              record_id={record_id}
              label={label}
              tagsForm={tagsForm}
              tagValues={tagValuesState}
              retrainAfterDecision={retrainAfterDecision}
              open={showTagsDialog}
              onClose={toggleShowTagsDialog}
              onSave={setTagValuesState}
            />
          )}
          {Array.isArray(listsForm) && listsForm.length > 0 && (
            <ListsDialog
              project_id={project_id}
              record_id={record_id}
              label={label}
              listsForm={listsForm}
              listValues={listValuesState}
              tagValues={tagValuesState}
              retrainAfterDecision={retrainAfterDecision}
              open={showListsDialog}
              onClose={toggleShowListsDialog}
              onSave={setListValuesState}
            />
          )}
        </CardActions>
      </Box>
    </Stack>
  );
};

export default RecordCardLabeler;
