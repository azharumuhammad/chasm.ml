import React from "react";

import { useTheme } from "@material-ui/core/styles";
import useMediaQuery from "@material-ui/core/useMediaQuery";

import Container from "@material-ui/core/Container";
import Grid from "@material-ui/core/Grid";
import Box from "@material-ui/core/Box";

export default function App() {
  const theme = useTheme();
  const matches = useMediaQuery(theme.breakpoints.up("md"));

  return (
    <Box p={matches ? 3 : 1}>
      <Container maxWidth="md">
        <Grid container>
          <Grid item md={3}></Grid>
          <Grid item md="auto">
            <h1>Hello World</h1>
          </Grid>
          <Grid item md={3}></Grid>
        </Grid>
      </Container>
    </Box>
  );
}
