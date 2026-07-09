import { Link as LinkIcon } from "@mui/icons-material";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Box,
  Button,
  Card,
  CardContent,
  Collapse,
  Divider,
  Fade,
  Grid2 as Grid,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
import React from "react";
import { useQuery } from "react-query";
import { useInView } from "react-intersection-observer";

import { ProjectAPI } from "api";
import { StyledIconButton } from "StyledComponents/StyledButton";
import { useToggle } from "hooks/useToggle";
import { useHighlightToggle } from "hooks/useHighlightToggle";
import { DOIIcon } from "icons";
import { LlmResultCard, RecordCardLabeler, RecordCardModelTraining } from ".";
import { useTheme } from "@mui/material/styles";
import {
  buildHighlightPatterns,
  highlightText,
  matchesAnyEntry,
  resolveColor,
} from "utils/highlight";

import { fontSizeOptions } from "globals.js";

const ZoteroFullTextButton = ({ project_id, record }) => {
  // Resolving the Zotero attachment requires a live API call per record. To avoid
  // firing one blocking request for every card at once (which monopolizes the server
  // threads and trips Zotero's rate limit), only check once the card scrolls into
  // view. `triggerOnce` keeps it checked afterwards.
  const { ref, inView } = useInView({ triggerOnce: true, rootMargin: "200px" });

  const { data } = useQuery(
    ["fetchRecordAttachment", { project_id, record_id: record?.record_id }],
    ProjectAPI.fetchRecordAttachment,
    {
      enabled: !!(
        inView &&
        project_id &&
        record?.record_id !== undefined &&
        record?.record_id !== null &&
        record?.original_id
      ),
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000,
      retry: false,
    },
  );

  // A diagonal line drawn across the PDF icon to mark "no full text available".
  const strikethrough = (
    <Box
      sx={{
        position: "absolute",
        top: "50%",
        left: "8%",
        width: "84%",
        height: "2px",
        bgcolor: "currentColor",
        borderRadius: 1,
        transform: "translateY(-50%) rotate(-45deg)",
      }}
    />
  );

  if (data?.available && data?.url) {
    return (
      <Tooltip title="Open full text in Zotero">
        <StyledIconButton
          ref={ref}
          className="record-card-icon"
          href={data.url}
          target="pdfreader"
        >
          <PictureAsPdfIcon />
        </StyledIconButton>
      </Tooltip>
    );
  }

  // Zotero is configured and confirmed there is no full text for this record: show a
  // struck-through PDF icon.
  if (data?.configured && data?.available === false) {
    return (
      <Tooltip title="No full text available in Zotero">
        <span ref={ref}>
          <StyledIconButton className="record-card-icon" disabled>
            <Box sx={{ position: "relative", display: "inline-flex" }}>
              <PictureAsPdfIcon />
              {strikethrough}
            </Box>
          </StyledIconButton>
        </span>
      </Tooltip>
    );
  }

  // Not checked yet (or Zotero not configured). Keep the observer target mounted so
  // the check still fires once the card scrolls into view.
  return <span ref={ref} />;
};

const RecordCardContent = ({
  project_id,
  record,
  fontSize,
  collapseAbstract,
  hideLinks = false,
  highlightEntries,
  highlightOn,
  paletteMode,
}) => {
  const [readMoreOpen, toggleReadMore] = useToggle();

  const renderText = (text) =>
    highlightOn && highlightEntries && highlightEntries.length > 0
      ? highlightText(text, highlightEntries, paletteMode)
      : text;

  return (
    <CardContent aria-label="record title abstract" sx={{ m: 1 }}>
      <Stack spacing={2}>
        {/* Show the title */}
        <Typography
          variant={"h5"}
          sx={(theme) => ({
            fontWeight: theme.typography.fontWeightMedium,
            lineHeight: 1.4,
          })}
        >
          {/* No title, inplace text */}
          {(record.title === "" || record.title === null) && (
            <Box
              className={"fontSize" + fontSizeOptions[fontSize]}
              fontStyle="italic"
            >
              No title available
            </Box>
          )}

          {!(record.title === "" || record.title === null) && (
            <Box className={"fontSize" + fontSizeOptions[fontSize]}>
              {renderText(record.title)}
            </Box>
          )}
        </Typography>
        <Divider />
        {!hideLinks && (
          <Stack direction="row" spacing={1}>
            {!(record.doi === undefined || record.doi === null) && (
              <Tooltip title="Open DOI">
                <StyledIconButton
                  className="record-card-icon"
                  href={"https://doi.org/" + record.doi}
                  target="_blank"
                  rel="noreferrer"
                >
                  <DOIIcon />
                </StyledIconButton>
              </Tooltip>
            )}

            {!(record.url === undefined || record.url === null) && (
              <Tooltip title="Open URL">
                <StyledIconButton
                  className="record-card-icon"
                  href={record.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  <LinkIcon />
                </StyledIconButton>
              </Tooltip>
            )}

            {!(
              record.original_id === undefined || record.original_id === null
            ) && (
              <ZoteroFullTextButton project_id={project_id} record={record} />
            )}

            {!(
              record.original_id === undefined || record.original_id === null
            ) && (
              <Typography
                variant="body2"
                sx={{ color: "text.secondary", alignSelf: "center" }}
              >
                ID: {record.original_id}
              </Typography>
            )}
          </Stack>
        )}
        <Box>
          {(record.abstract === "" || record.abstract === null) && (
            <Typography
              className={"fontSize" + fontSize}
              variant="body1"
              sx={{
                fontStyle: "italic",
                textAlign: "justify",
              }}
            >
              No abstract available
            </Typography>
          )}

          <Typography
            className={"fontSize" + fontSizeOptions[fontSize]}
            variant="body1"
            sx={{
              whiteSpace: "pre-line",
              textAlign: "justify",
              hyphens: "auto",
              lineHeight: 1.6,
            }}
          >
            {!(record.abstract === "" || record.abstract === null) &&
            collapseAbstract &&
            record.abstract.length > 500 ? (
              <>
                {!readMoreOpen ? (
                  <>
                    {renderText(record.abstract.substring(0, 500))}...
                    <Button
                      onClick={toggleReadMore}
                      startIcon={<ExpandMoreIcon />}
                      color="primary"
                      sx={{ textTransform: "none" }}
                    >
                      show more
                    </Button>
                  </>
                ) : (
                  <>
                    {renderText(record.abstract)}
                    <Button
                      onClick={toggleReadMore}
                      startIcon={<ExpandLessIcon />}
                      color="primary"
                      sx={{ textTransform: "none" }}
                    >
                      show less
                    </Button>
                  </>
                )}
              </>
            ) : (
              renderText(record.abstract)
            )}
          </Typography>
        </Box>
        {record.keywords && (
          <Box sx={{ pt: 1 }}>
            <Typography sx={{ color: "text.secondary", fontWeight: "bold" }}>
              {record.keywords.map((keyword, index) => {
                const matched =
                  highlightOn && highlightEntries
                    ? matchesAnyEntry(keyword, highlightEntries)
                    : null;
                const bg = matched
                  ? resolveColor(matched.color, paletteMode)
                  : null;
                return (
                  <span key={index}>
                    {index > 0 && " • "}
                    {bg ? (
                      <span
                        style={{
                          backgroundColor: bg,
                          padding: "0 4px",
                          borderRadius: 3,
                        }}
                      >
                        {keyword}
                      </span>
                    ) : (
                      keyword
                    )}
                  </span>
                );
              })}
            </Typography>
          </Box>
        )}
      </Stack>
    </CardContent>
  );
};

const RecordCard = ({
  project_id,
  record,
  afterDecision = null,
  onDiscarded = null,
  retrainAfterDecision = true,
  showBorder = true,
  fontSize = 1,
  modelLogLevel = "warning",
  showNotes = true,
  collapseAbstract = false,
  hideLinks = false,
  hotkeys = false,
  transitionType = "fade",
  transitionSpeed = { enter: 500, exit: 100 },
  landscape = false,
  changeDecision = true,
}) => {
  const [open, setOpen] = React.useState(true);
  const theme = useTheme();
  const [highlightOn, toggleHighlight] = useHighlightToggle();
  const [isDirty, setIsDirty] = React.useState(false);
  const [resetKey, setResetKey] = React.useState(0);
  const [llmTagValues, setLlmTagValues] = React.useState(null);
  const [llmListValues, setLlmListValues] = React.useState(null);

  const handleApplyLlm = () => {
    ProjectAPI.applyLlm({ project_id, record_id: record.record_id }).then(
      (data) => {
        setLlmTagValues(data.tags);
        setLlmListValues(data.lists);
        setResetKey((k) => k + 1);
      },
    );
  };

  // Reset LLM-applied state when the record changes.
  React.useEffect(() => {
    setLlmTagValues(null);
    setLlmListValues(null);
    setResetKey(0);
    setIsDirty(false);
  }, [record.record_id]);

  const displayTagValues = llmTagValues || record.state?.tags;
  const displayListValues = llmListValues || record.state?.lists;

  const { data: highlightConfig } = useQuery(
    ["fetchHighlights", { project_id }],
    ProjectAPI.fetchHighlights,
    {
      enabled: !!project_id,
      refetchOnWindowFocus: false,
      staleTime: 60 * 1000,
    },
  );

  const highlightAvailable = !!(
    highlightConfig &&
    Array.isArray(highlightConfig.groups) &&
    highlightConfig.groups.some(
      (g) => Array.isArray(g.words) && g.words.length > 0,
    )
  );

  const highlightEntries = React.useMemo(
    () =>
      highlightAvailable ? buildHighlightPatterns(highlightConfig.groups) : [],
    [highlightAvailable, highlightConfig],
  );

  const styledRepoCard = (
    <Box>
      <RecordCardModelTraining
        key={"record-card-model-" + project_id + "-" + record?.record_id}
        record={record}
        modelLogLevel={modelLogLevel}
        sx={{ mb: 3 }}
      />
      <Card
        elevation={showBorder ? 4 : 0}
        sx={(theme) => ({
          bgcolor: theme.palette.background.record,
          borderRadius: !showBorder ? 0 : undefined,
        })}
      >
        <Grid
          container
          columns={5}
          sx={{ alignItems: "stretch" }}
          // divider={<Divider orientation="vertical" flexItem />}
        >
          <Grid size={landscape ? 3 : 5}>
            <RecordCardContent
              project_id={project_id}
              record={record}
              fontSize={fontSize}
              collapseAbstract={collapseAbstract}
              hideLinks={hideLinks}
              highlightEntries={highlightEntries}
              highlightOn={highlightOn && highlightAvailable}
              paletteMode={theme.palette.mode}
            />
            <LlmResultCard
              project_id={project_id}
              record_id={record.record_id}
              llm={record.llm}
              isDirty={isDirty}
              onApplyLlm={handleApplyLlm}
            />
          </Grid>
          <Grid size={landscape ? 2 : 5}>
            <RecordCardLabeler
              key={
                "record-card-labeler-" + project_id + "-" + record?.record_id
              }
              project_id={project_id}
              record_id={record.record_id}
              label={record.state?.label}
              labelFromDataset={record.included}
              onDecisionClose={
                transitionType ? () => setOpen(false) : afterDecision
              }
              onDiscarded={onDiscarded}
              retrainAfterDecision={retrainAfterDecision}
              note={record.state?.note}
              labelTime={record.state?.time}
              user={record.state?.user}
              showNotes={showNotes}
              tagsForm={record.tags_form}
              tagValues={displayTagValues}
              listsForm={record.lists_form}
              listValues={displayListValues}
              landscape={landscape}
              hotkeys={hotkeys}
              changeDecision={changeDecision}
              highlightAvailable={highlightAvailable}
              highlightOn={highlightOn}
              onToggleHighlight={toggleHighlight}
              resetKey={resetKey}
              onDirtyChange={setIsDirty}
            />
          </Grid>
        </Grid>
      </Card>
    </Box>
  );

  if (transitionType === "fade") {
    return (
      <Fade
        in={open}
        timeout={transitionSpeed}
        onExited={afterDecision}
        unmountOnExit
      >
        {styledRepoCard}
      </Fade>
    );
  } else if (transitionType === "collapse") {
    return (
      <Collapse
        in={open}
        timeout={transitionSpeed}
        onExited={afterDecision}
        unmountOnExit
      >
        {styledRepoCard}
      </Collapse>
    );
  } else {
    return styledRepoCard;
  }
};

export default RecordCard;
