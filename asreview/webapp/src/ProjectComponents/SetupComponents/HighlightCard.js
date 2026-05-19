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
  IconButton,
  MenuItem,
  Select,
  Skeleton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import { useMutation, useQuery, useQueryClient } from "react-query";

import { useTheme } from "@mui/material/styles";

import { ProjectAPI } from "api";
import { ProjectContext } from "context/ProjectContext";
import { HIGHLIGHT_COLORS, resolveColor } from "utils/highlight";

const colorLabel = (value) =>
  HIGHLIGHT_COLORS.find((c) => c.value === value)?.label || value;

const parseWords = (text) =>
  text
    .split(/[,\n]/)
    .map((w) => w.trim())
    .filter(Boolean);

const Swatch = ({ color, size = 18 }) => {
  const theme = useTheme();
  return (
    <Box
      sx={{
        width: size,
        height: size,
        borderRadius: "50%",
        backgroundColor: resolveColor(color, theme.palette.mode),
        border: 1,
        borderColor: "divider",
        flexShrink: 0,
      }}
    />
  );
};

const EditDialog = ({ open, onClose, initialGroups, onSave }) => {
  // Each row: { id, color, text } — text is the raw textarea contents.
  const [rows, setRows] = React.useState([]);

  React.useEffect(() => {
    if (!open) return;
    setRows(
      (initialGroups || []).map((g, i) => ({
        id: g.id ?? i + 1,
        color: g.color,
        text: (g.words || []).join(", "),
      })),
    );
  }, [open, initialGroups]);

  const usedColors = new Set(rows.map((r) => r.color));
  const availableColors = HIGHLIGHT_COLORS.filter(
    (c) => !usedColors.has(c.value),
  );

  const updateRow = (id, patch) =>
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));

  const removeRow = (id) => setRows((rs) => rs.filter((r) => r.id !== id));

  const addRow = () => {
    if (availableColors.length === 0) return;
    setRows((rs) => [
      ...rs,
      {
        id: Math.max(0, ...rs.map((r) => r.id)) + 1,
        color: availableColors[0].value,
        text: "",
      },
    ]);
  };

  // Validation: a row's color must be one of the 6, and colors must be unique.
  // (Empty word lists are allowed — silently dropped on save.)
  const colorCounts = rows.reduce((acc, r) => {
    acc[r.color] = (acc[r.color] || 0) + 1;
    return acc;
  }, {});
  const hasDuplicateColor = Object.values(colorCounts).some((n) => n > 1);
  const hasInvalidColor = rows.some(
    (r) => !HIGHLIGHT_COLORS.find((c) => c.value === r.color),
  );
  const isInvalid = hasDuplicateColor || hasInvalidColor;

  const handleSave = () => {
    if (isInvalid) return;
    const groups = rows
      .map((r) => ({
        id: r.id,
        color: r.color,
        words: parseWords(r.text),
      }))
      .filter((g) => g.words.length > 0);
    onSave(groups);
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Edit highlight words</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <Alert severity="info">
            Separate words with commas. Use <code>*</code> at the end of a word
            for open word endings (e.g. <code>immun*</code> matches "immune",
            "immunology"). Multi-word phrases are allowed (e.g.{" "}
            <code>randomized controlled trial</code>). Longest match wins. Empty
            colors are dropped on save. The feature is automatically disabled if
            no words are configured.
          </Alert>

          {rows.map((row) => (
            <Box
              key={row.id}
              sx={{
                p: 2,
                border: 1,
                borderColor: "divider",
                borderRadius: 2,
              }}
            >
              <Stack direction="row" spacing={2} alignItems="flex-start">
                <Select
                  size="small"
                  value={row.color}
                  onChange={(e) => updateRow(row.id, { color: e.target.value })}
                  renderValue={(v) => (
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Swatch color={v} />
                      <span>{colorLabel(v)}</span>
                    </Stack>
                  )}
                  sx={{ minWidth: 140 }}
                >
                  {HIGHLIGHT_COLORS.map((c) => (
                    <MenuItem
                      key={c.value}
                      value={c.value}
                      disabled={
                        c.value !== row.color && usedColors.has(c.value)
                      }
                    >
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Swatch color={c.value} />
                        <span>{c.label}</span>
                      </Stack>
                    </MenuItem>
                  ))}
                </Select>
                <TextField
                  multiline
                  minRows={2}
                  fullWidth
                  size="small"
                  label="Words / phrases"
                  placeholder="e.g. randomized, immun*, controlled trial"
                  value={row.text}
                  onChange={(e) => updateRow(row.id, { text: e.target.value })}
                />
                <IconButton
                  onClick={() => removeRow(row.id)}
                  aria-label="remove color"
                >
                  <DeleteIcon />
                </IconButton>
              </Stack>
            </Box>
          ))}

          <Box>
            <Button
              startIcon={<AddIcon />}
              onClick={addRow}
              disabled={availableColors.length === 0}
            >
              Add color
            </Button>
          </Box>

          {hasDuplicateColor && (
            <Alert severity="error">
              Each color can only be used once. Pick a different color for the
              duplicates.
            </Alert>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} disabled={isInvalid} variant="contained">
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const HighlightCard = ({ project_id: project_id_prop, editable = true }) => {
  const queryClient = useQueryClient();
  const ctxProjectId = React.useContext(ProjectContext);
  const project_id = project_id_prop || ctxProjectId;
  const theme = useTheme();

  const [editOpen, setEditOpen] = React.useState(false);

  const { data, isLoading } = useQuery(
    ["fetchHighlights", { project_id }],
    ProjectAPI.fetchHighlights,
    {
      enabled: !!project_id,
      refetchOnWindowFocus: false,
    },
  );

  const { mutate: save } = useMutation(ProjectAPI.mutateHighlights, {
    onSuccess: () => {
      queryClient.invalidateQueries(["fetchHighlights", { project_id }]);
    },
    onError: (err) => console.error("Failed to save highlights:", err),
  });

  const groups = data?.groups || [];
  const featureActive = groups.some(
    (g) => Array.isArray(g.words) && g.words.length > 0,
  );

  if (isLoading) {
    return (
      <Card>
        <CardHeader title="Highlight words" />
        <CardContent>
          <Skeleton variant="rounded" height={120} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title="Highlight words"
        subheader={
          featureActive
            ? "Words below are highlighted in title, abstract and record keywords during screening."
            : "No words configured — highlighting is disabled. Click Edit to add some."
        }
        action={
          editable && (
            <Tooltip title="Edit highlight words">
              <IconButton
                onClick={() => setEditOpen(true)}
                aria-label="edit highlight words"
              >
                <EditIcon />
              </IconButton>
            </Tooltip>
          )
        }
      />
      <CardContent>
        {groups.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No highlight words configured.
          </Typography>
        ) : (
          <Stack spacing={1.5}>
            {groups.map((g) => (
              <Stack
                key={g.id}
                direction="row"
                spacing={1}
                alignItems="center"
                flexWrap="wrap"
                useFlexGap
              >
                <Swatch color={g.color} />
                <Typography
                  variant="subtitle2"
                  sx={{ minWidth: 64, color: "text.secondary" }}
                >
                  {colorLabel(g.color)}
                </Typography>
                {(g.words || []).map((w, i) => (
                  <Chip
                    key={i}
                    label={w}
                    size="small"
                    sx={{
                      backgroundColor: resolveColor(
                        g.color,
                        theme.palette.mode,
                      ),
                    }}
                  />
                ))}
              </Stack>
            ))}
          </Stack>
        )}
      </CardContent>

      {editable && (
        <EditDialog
          open={editOpen}
          onClose={() => setEditOpen(false)}
          initialGroups={groups}
          onSave={(newGroups) => {
            save({ project_id, config: { groups: newGroups } });
            setEditOpen(false);
          }}
        />
      )}
    </Card>
  );
};

export default HighlightCard;
