name         := "Sort"
scalaVersion := "2.13.12"
version      := "1.0"

val chiselVersion = "6.0.0"

lazy val root = (project in file("."))
  .settings(
    name := "Sort",
    libraryDependencies += "org.chipsalliance" %% "chisel" % chiselVersion,
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full)
  )


