import React from "react";
import {
  Alert,
  Box,
  Typography,
  Checkbox,
  Radio,
  FormGroup,
  FormControlLabel,
  TextField,
} from "@mui/material";

const TagsTable = ({
  tagsForm,
  setTagValues,
  tagValues = null,
  disabled = false,
}) => {
  const handleTagValueChange = (isChecked, groupId, tagId) => {
    let groupI = tagValues.findIndex((group) => group.id === groupId);
    let tagI = tagValues[groupI].values.findIndex((tag) => tag.id === tagId);

    let tagValuesCopy = tagValues;
    tagValuesCopy[groupI].values[tagI]["checked"] = isChecked;

    setTagValues(tagValuesCopy);
  };

  const handleSingleSelect = (groupId, tagId) => {
    let groupI = tagValues.findIndex((group) => group.id === groupId);

    let tagValuesCopy = tagValues;
    tagValuesCopy[groupI].values = tagValuesCopy[groupI].values.map((tag) => ({
      ...tag,
      checked: tag.id === tagId,
    }));

    setTagValues(tagValuesCopy);
  };

  const handleTagTextChange = (groupId, tagId, text) => {
    let groupI = tagValues.findIndex((group) => group.id === groupId);
    let tagI = tagValues[groupI].values.findIndex((tag) => tag.id === tagId);

    let tagValuesCopy = tagValues;
    tagValuesCopy[groupI].values[tagI]["text"] = text;

    setTagValues(tagValuesCopy);
  };

  return (
    <>
      {tagsForm &&
        tagsForm.map((group, i) => {
          const singleSelect = Boolean(group.single_select);
          const required = Boolean(
            group.required_relevant || group.required_irrelevant,
          );
          const checkedCount = (tagValues[i]?.values || []).filter(
            (t) => t.checked,
          ).length;
          const invalid = singleSelect && checkedCount > 1;
          const missing = required && checkedCount === 0;
          return (
            <Box key={group.id}>
              <Typography variant="h6">
                {group.name}
                {required && " *"}
              </Typography>
              {group.input_helper_text && (
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ fontStyle: "italic", mb: 0.5 }}
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
                <Typography variant="caption" color="error">
                  {singleSelect
                    ? "Select one option"
                    : "Select at least one option"}
                </Typography>
              )}
              <FormGroup row={true}>
                {group.values.map((tag, j) => {
                  const checked = tagValues[i]?.values[j]?.checked || false;
                  const text = tagValues[i]?.values[j]?.text || "";
                  return (
                    <Box key={`${group.id}:${tag.id}`}>
                      <FormControlLabel
                        control={
                          singleSelect ? (
                            <Radio
                              checked={checked}
                              onChange={() =>
                                handleSingleSelect(group.id, tag.id)
                              }
                              onClick={() => {
                                // Radio buttons can always be deselected.
                                if (checked) {
                                  handleTagValueChange(false, group.id, tag.id);
                                }
                              }}
                              disabled={disabled}
                            />
                          ) : (
                            <Checkbox
                              checked={checked}
                              onChange={(e) =>
                                handleTagValueChange(
                                  e.target.checked,
                                  group.id,
                                  tag.id,
                                )
                              }
                              disabled={disabled}
                            />
                          )
                        }
                        label={tag.name}
                      />
                      {tag.free_text && checked && (
                        <TextField
                          size="small"
                          fullWidth
                          variant="standard"
                          placeholder="Add free text…"
                          value={text}
                          onChange={(e) =>
                            handleTagTextChange(
                              group.id,
                              tag.id,
                              e.target.value,
                            )
                          }
                          disabled={disabled}
                          sx={{ ml: 4, mb: 1 }}
                        />
                      )}
                    </Box>
                  );
                })}
              </FormGroup>
            </Box>
          );
        })}
    </>
  );
};

export default TagsTable;
