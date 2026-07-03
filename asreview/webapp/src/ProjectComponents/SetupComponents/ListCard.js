import React from "react";

import {
  Alert,
  Button,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  Skeleton,
  Stack,
  Switch,
  TextField,
  Tooltip,
} from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import { ProjectContext } from "context/ProjectContext";
import { useContext } from "react";
import { LoadingCardHeader } from "StyledComponents/LoadingCardheader";

import { ProjectAPI } from "api";
import { useMutation, useQuery, useQueryClient } from "react-query";

import { useTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";

import { useToggle } from "hooks/useToggle";

const MutateListDialog = ({ project_id, open, onClose, list = null }) => {
  const theme = useTheme();
  const queryClient = useQueryClient();
  const smallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  const [name, setName] = React.useState(list ? list.name : "");
  const [inputHelperText, setInputHelperText] = React.useState(
    list ? list.input_helper_text || "" : "",
  );
  const [requiredForRelevant, setRequiredForRelevant] = React.useState(
    list ? Boolean(list.required_for_relevant) : false,
  );

  React.useEffect(() => {
    if (open) {
      setName(list ? list.name : "");
      setInputHelperText(list ? list.input_helper_text || "" : "");
      setRequiredForRelevant(
        list ? Boolean(list.required_for_relevant) : false,
      );
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const { mutate: createList, error: createError } = useMutation(
    ProjectAPI.createList,
    {
      mutationKey: ["createList"],
      onSuccess: () => {
        queryClient.invalidateQueries(["fetchLists", { project_id }]);
        onClose();
      },
    },
  );

  const { mutate: mutateList, error: mutateError } = useMutation(
    ProjectAPI.mutateList,
    {
      mutationKey: ["mutateList"],
      onSuccess: () => {
        queryClient.invalidateQueries(["fetchLists", { project_id }]);
        onClose();
      },
    },
  );

  const onSave = () => {
    if (list !== null) {
      mutateList({
        project_id,
        list: {
          id: list.id,
          name,
          input_helper_text: inputHelperText,
          required_for_relevant: requiredForRelevant,
        },
      });
    } else {
      createList({
        project_id,
        list: {
          name,
          input_helper_text: inputHelperText,
          required_for_relevant: requiredForRelevant,
        },
      });
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullScreen={smallScreen}
      fullWidth
      maxWidth="sm"
    >
      <DialogTitle>{list !== null ? "Edit list" : "Add list"}</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ pt: 1 }}>
          <TextField
            fullWidth
            id="list-name"
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <TextField
            fullWidth
            id="list-input-helper-text"
            label="Input helper text"
            value={inputHelperText}
            onChange={(e) => setInputHelperText(e.target.value)}
            helperText="Optional help text shown below the list header during editing"
          />
          <Tooltip title="Require at least one item before a record can be marked relevant">
            <FormControlLabel
              control={
                <Switch
                  checked={requiredForRelevant}
                  onChange={(e) => setRequiredForRelevant(e.target.checked)}
                />
              }
              label="Require an item to mark a record as relevant"
            />
          </Tooltip>
          {(createError || mutateError) && (
            <Alert severity="error">
              {createError?.message || mutateError?.message}
            </Alert>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={onSave} disabled={!name}>
          {list !== null ? "Save" : "Create List"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const ListItem = ({ project_id, list }) => {
  const [dialogOpen, toggleDialogOpen] = useToggle();

  return (
    <>
      <Chip
        label={`${list.name}${list.required_for_relevant ? " *" : ""}`}
        onDelete={toggleDialogOpen}
        deleteIcon={
          <Tooltip title="Edit list">
            <EditIcon />
          </Tooltip>
        }
        sx={{ m: 1 }}
      />
      <MutateListDialog
        key={list.id}
        project_id={project_id}
        open={dialogOpen}
        onClose={toggleDialogOpen}
        list={list}
      />
    </>
  );
};

const ListCard = () => {
  const project_id = useContext(ProjectContext);
  const [dialogOpen, toggleDialogOpen] = useToggle();

  const { data, isLoading } = useQuery(
    ["fetchLists", { project_id: project_id }],
    ProjectAPI.fetchLists,
    {
      refetchOnWindowFocus: false,
    },
  );

  return (
    <Card>
      <LoadingCardHeader
        title="Lists"
        subheader="Lists let reviewers add several free-text items per record (for example extracted outcomes or populations)"
        isLoading={isLoading}
      />

      <CardContent>
        {isLoading ? (
          <Skeleton variant="rectangular" height={56} />
        ) : (
          <>
            {data.length === 0 && (
              <Alert severity="info" sx={{ mb: 2 }}>
                Your lists will appear here
              </Alert>
            )}
            {data.map((list) => (
              <ListItem key={list.id} list={list} project_id={project_id} />
            ))}
          </>
        )}
      </CardContent>

      <CardContent>
        {isLoading ? (
          <Skeleton variant="rectangular" width={100} height={36} />
        ) : (
          <>
            <MutateListDialog
              project_id={project_id}
              open={dialogOpen}
              onClose={toggleDialogOpen}
            />
            <Button onClick={toggleDialogOpen} variant="contained">
              Add list
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ListCard;
