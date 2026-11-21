import { Box, ButtonBase, Fade, Stack, Typography } from "@mui/material";
import { grey } from "@mui/material/colors";
import React from "react";
import { InView } from "react-intersection-observer";
import { useInfiniteQuery, useQuery, useQueryClient } from "react-query";

import { InlineErrorHandler } from "Components";
import { RecordCard } from "ProjectComponents/ReviewComponents";
import { ProjectAPI } from "api";

import { useMediaQuery } from "@mui/material";
import { useReviewSettings } from "context/ReviewSettingsContext";

import {
  formsOf,
  getNextPageParam,
  patchRowInPages,
  usersOf,
} from "./labeledPaging";

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

  const queryClient = useQueryClient();
  const filterValues = filterQuery.map((filter) => filter.value);

  // The query key is memoised on the filter VALUES (JSON), not the array
  // identity, so a parent re-render with the same filters keeps the same key
  // and `setQueryData` patching (below) targets the right query.
  const queryKey = React.useMemo(
    () => [
      "fetchLabeledRecord",
      { project_id: project_id, subset: label, filter: filterValues },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [project_id, label, JSON.stringify(filterValues)],
  );

  const {
    data,
    error,
    fetchNextPage,
    hasNextPage,
    isError,
    isFetched,
    isFetchingNextPage,
    isLoading,
  } = useInfiniteQuery(queryKey, ProjectAPI.fetchLabeledRecord, {
    getNextPageParam,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: "always",
    cacheTime: 0, // no snapshot reuse after navigating away
  });

  // Patch exactly one row (an own edit) instead of refetching every page.
  const onRowUpdated = React.useCallback(
    (row) => {
      if (!row) return;
      queryClient.setQueryData(queryKey, (old) =>
        old ? patchRowInPages(old, row) : old,
      );
      queryClient.invalidateQueries(["fetchLabeledStats", { project_id }]);
    },
    [queryClient, queryKey, project_id],
  );

  const forms = formsOf(data);
  const users = usersOf(data);

  /**
   * Check if this component is mounted
   */
  const mounted = React.useRef(false);
  React.useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  return (
    <Box aria-label="labeled record">
      {isError && <InlineErrorHandler message={error?.message} />}
      {/* {n_prior !== 0 && !isError && (isLoading || !mounted.current) && (
        <Box className={classes.loading}>
          <CircularProgress />
        </Box>
      )} */}
      {!isError && !(isLoading || !mounted.current) && isFetched && (
        <Fade in={!isError && !(isLoading || !mounted.current) && isFetched}>
          <Stack aria-label="labeled record card" spacing={5}>
            {isFetched &&
              data?.pages.map((page) =>
                page.result.map((record) => (
                  <RecordCard
                    project_id={project_id}
                    record={record}
                    collapseAbstract={!expandAbstract}
                    disabled={true}
                    isOwner={isOwner}
                    allowMemberReplace={
                      projectInfo?.allow_member_replace === true
                    }
                    transitionType={null}
                    landscape={
                      orientation === "landscape" && !landscapeDisabled
                    }
                    modelLogLevel={modelLogLevel}
                    changeDecision={mode === "oracle"}
                    tagsForm={forms.tags_form}
                    listsForm={forms.lists_form}
                    users={users}
                    onRowUpdated={onRowUpdated}
                    key={"record-card-" + project_id + "-" + record?.record_id}
                  />
                )),
              )}
            <Box sx={{ display: "flex", justifyContent: "center" }}>
              <InView
                as="div"
                onChange={(inView, entry) => {
                  if (inView && hasNextPage && !isFetchingNextPage) {
                    fetchNextPage();
                  }
                }}
              >
                <ButtonBase disabled={!hasNextPage || isFetchingNextPage}>
                  <Typography
                    gutterBottom
                    variant="button"
                    sx={{ color: grey[500] }}
                  >
                    {isFetchingNextPage
                      ? "Loading more..."
                      : hasNextPage
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
