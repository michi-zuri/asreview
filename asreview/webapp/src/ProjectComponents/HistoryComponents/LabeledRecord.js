import {
  Alert,
  Box,
  Button,
  ButtonBase,
  CircularProgress,
  Fade,
  Stack,
  Typography,
  useMediaQuery,
} from "@mui/material";
import { grey } from "@mui/material/colors";
import React from "react";
import { InView } from "react-intersection-observer";
import { useQuery } from "react-query";

import { InlineErrorHandler } from "Components";
import { RecordCard } from "ProjectComponents/ReviewComponents";
import { ProjectAPI } from "api";
import { useReviewSettings } from "context/ReviewSettingsContext";

import { visibleRows } from "./collectionModel";
import useCollection from "./useCollection";

const LabeledRecord = ({ project_id, label, filterQuery, mode = "oracle" }) => {
  const { orientation, modelLogLevel, expandAbstract } = useReviewSettings();

  const { data: projectInfo } = useQuery(
    ["fetchProject", { project_id }],
    ProjectAPI.fetchInfo,
    {
      refetchOnWindowFocus: false,
    },
  );
  const isOwner = projectInfo?.roles?.owner === true;

  let landscapeDisabled = useMediaQuery(
    (theme) => theme.breakpoints.down("md"),
    {
      noSsr: true,
    },
  );

  // Memoise on the filter VALUES so the snapshot key (and therefore the poll
  // interval) stays stable across parent re-renders.
  const filterValues = React.useMemo(
    () => filterQuery.map((filter) => filter.value),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(filterQuery)],
  );

  const {
    state,
    isLoading,
    isLoadingMore,
    error,
    loadMore,
    refresh,
    onRowUpdated,
    onEditingChange,
  } = useCollection({ project_id, subset: label, filter: filterValues });

  const rows = visibleRows(state);

  return (
    <Box aria-label="labeled record">
      {error && <InlineErrorHandler message={error} />}
      {isLoading ? (
        <Box sx={{ display: "flex", justifyContent: "center", pt: 5 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Fade in={!isLoading}>
          <Stack aria-label="labeled record card" spacing={5}>
            {state.stale && (
              <Alert
                severity="info"
                action={
                  <Button color="inherit" size="small" onClick={refresh}>
                    Refresh
                  </Button>
                }
                sx={{ position: "sticky", top: 0, zIndex: 2 }}
              >
                This view is out of date.
              </Alert>
            )}
            {rows.map((record) => (
              <RecordCard
                project_id={project_id}
                record={record}
                collapseAbstract={!expandAbstract}
                disabled={true}
                isOwner={isOwner}
                allowMemberReplace={projectInfo?.allow_member_replace === true}
                transitionType={null}
                landscape={orientation === "landscape" && !landscapeDisabled}
                modelLogLevel={modelLogLevel}
                changeDecision={mode === "oracle"}
                tagsForm={state.forms.tags_form}
                listsForm={state.forms.lists_form}
                users={state.users}
                onRowUpdated={onRowUpdated}
                onEditingChange={onEditingChange}
                key={"record-card-" + project_id + "-" + record?.record_id}
              />
            ))}
            <Box sx={{ display: "flex", justifyContent: "center" }}>
              <InView
                as="div"
                onChange={(inView) => {
                  if (inView && state.hasMore) {
                    loadMore();
                  }
                }}
              >
                <ButtonBase disabled={!state.hasMore || isLoadingMore}>
                  <Typography
                    gutterBottom
                    variant="button"
                    sx={{ color: grey[500] }}
                  >
                    {isLoadingMore
                      ? "Loading more..."
                      : state.hasMore
                        ? "Load More"
                        : "Nothing more to load"}
                  </Typography>
                </ButtonBase>
              </InView>
            </Box>
          </Stack>
        </Fade>
      )}
    </Box>
  );
};

export default LabeledRecord;
