import * as React from "react";
import ReactLoading from "react-loading";
import { useMutation, useQuery } from "react-query";
import { ProjectAPI } from "api";

import {
  Button,
  Stack,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Fade,
  LinearProgress,
  Box,
} from "@mui/material";
import { styled, useTheme } from "@mui/material/styles";

import { useToggle } from "hooks/useToggle";
import ElasPad from "images/ElasPad.svg";

// const YouTubeVideoID = "k-a2SCq-LtA";

const PREFIX = "ReviewPageTraining";

const classes = {
  img: `${PREFIX}-img`,
  textTitle: `${PREFIX}-textTitle`,
  text: `${PREFIX}-text`,
};

const Root = styled("div")(({ theme }) => ({
  height: "inherit",
  [`& .${classes.img}`]: {
    maxWidth: 350,
    [theme.breakpoints.down("md")]: {
      maxWidth: 250,
    },
  },
  [`& .${classes.textTitle}`]: {
    textAlign: "center",
    [theme.breakpoints.down("md")]: {
      width: "80%",
    },
  },
  [`& .${classes.text}`]: {
    textAlign: "center",
    width: "60%",
    [theme.breakpoints.down("md")]: {
      width: "80%",
    },
  },
}));

const TRAINING_PHASES = [
  "loading",
  "feature_extraction",
  "fitting",
  "ranking",
  "saving",
];

const phaseToStep = (phase) => {
  const idx = TRAINING_PHASES.indexOf(phase);
  return idx >= 0 ? idx : 0;
};

// Seconds after which we consider training to not be running
// and show the retry button
const TRAINING_STALE_SECONDS = 15;

const FinishSetup = ({ project_id, refetch }) => {
  const theme = useTheme();

  const [openSkipTraining, toggleSkipTraining] = useToggle();
  const autoTriggered = React.useRef(false);

  // Poll training progress every 3 seconds
  const { data: progress } = useQuery(
    ["fetchTrainingProgress", { project_id }],
    ProjectAPI.fetchTrainingProgress,
    {
      refetchInterval: 500,
      refetchIntervalInBackground: true,
      refetchOnWindowFocus: false,
    },
  );

  // mutate and start new training
  const {
    mutate: startTraining,
    isLoading: isTraining,
    isError: isTrainingError,
  } = useMutation(ProjectAPI.mutateTraining, {
    onSuccess: () => {
      refetch();
    },
  });

  // Auto-trigger training on mount. This is a safety net: if the
  // setup->review transition failed to start training, this ensures
  // it gets started. The backend is idempotent — duplicate queue
  // entries for the same project are rejected by the task manager.
  React.useEffect(() => {
    if (autoTriggered.current) return;
    autoTriggered.current = true;
    startTraining({ project_id: project_id });
  }, [project_id, startTraining]);

  const retryTraining = () => {
    autoTriggered.current = true;
    startTraining({ project_id: project_id });
  };

  const skipTraining = (method) => {
    if (method === "random") {
      startTraining({ project_id: project_id, ranking: "random" });
    } else if (method === "top_down") {
      startTraining({ project_id: project_id, ranking: "top_down" });
    }
  };

  const currentStep = progress?.phase ? phaseToStep(progress.phase) : 0;
  const progressPercent = ((currentStep + 1) / TRAINING_PHASES.length) * 100;

  const progressLabel = progress?.label || "Warming up the AI!";

  const datasetInfo =
    progress?.n_records && progress?.n_labeled
      ? `${progress.n_labeled.toLocaleString()} labeled records out of ${progress.n_records.toLocaleString()}`
      : null;

  // Show retry if we auto-triggered but still no progress after a while,
  // or if the training mutation returned an error
  const showRetry =
    isTrainingError ||
    (autoTriggered.current && !isTraining && !progress?.phase);

  return (
    <Root aria-label="review page training">
      <Fade in>
        <Stack
          spacing={1}
          sx={{
            alignItems: "center",
            height: "inherit",
            justifyContent: "center",
          }}
        >
          <img src={ElasPad} alt="ElasPad" className={classes.img} />
          <Typography className={classes.textTitle} variant="h5">
            Warming up the AI!
          </Typography>
          {progress?.phase ? (
            <Box sx={{ width: "60%", maxWidth: 400 }}>
              <LinearProgress
                variant="determinate"
                value={progressPercent}
                sx={{ height: 8, borderRadius: 4 }}
              />
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mt: 1, textAlign: "center" }}
              >
                {progressLabel}
              </Typography>
              {datasetInfo && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ textAlign: "center", display: "block" }}
                >
                  {datasetInfo}
                </Typography>
              )}
            </Box>
          ) : (
            <ReactLoading
              type="bubbles"
              color={theme.palette.primary.main}
              height={60}
              width={60}
            />
          )}
          {showRetry && (
            <Button
              variant="outlined"
              onClick={retryTraining}
              disabled={isTraining}
            >
              Retry training
            </Button>
          )}
          <Button onClick={toggleSkipTraining} disabled={isTraining}>
            I can't wait
          </Button>
        </Stack>
      </Fade>
      {/* {isError && (
      )} */}
      <Dialog
        open={openSkipTraining}
        onClose={toggleSkipTraining}
        aria-labelledby="skip-training-dialog"
      >
        <DialogTitle id="skip-training-dialog">
          Review already? Let's get started!
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            Do you want to review already? Your model will be trained in the
            background. Once the model is finished training, you see better
            results. Choose one of the following options to start reviewing:
          </DialogContentText>
          <Button onClick={() => skipTraining("random")} disabled={isTraining}>
            Random
          </Button>
          <Button
            onClick={() => skipTraining("top_down")}
            disabled={isTraining}
          >
            Top down
          </Button>
        </DialogContent>
        <DialogActions>
          <Button onClick={toggleSkipTraining}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Root>
  );
};

export default FinishSetup;
