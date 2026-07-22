import { defineConfig } from "vitepress";

export default defineConfig({
  title: "praxis-eval",
  description: "Standalone robot-policy evaluation documentation",
  base: "/praxis-eval/",
  cleanUrls: false,
  appearance: true,
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "Quickstart", link: "/quickstart" },
      { text: "Benchmarks", link: "/benchmarks/libero" },
      { text: "Development", link: "/development/adding-benchmarks" },
      { text: "GitHub", link: "https://github.com/Chaoqi-LIU/praxis-eval" },
    ],
    sidebar: [
      {
        text: "Getting Started",
        items: [
          { text: "Quickstart", link: "/quickstart" },
          { text: "Installation", link: "/installation" },
        ],
      },
      {
        text: "Concepts",
        items: [
          { text: "Evaluation Loop", link: "/concepts/evaluation-loop" },
          {
            text: "Observations And Actions",
            link: "/concepts/observations-and-actions",
          },
          {
            text: "Local Vs Remote Policy",
            link: "/concepts/local-vs-remote-policy",
          },
          {
            text: "Results And Artifacts",
            link: "/concepts/results-and-artifacts",
          },
        ],
      },
      {
        text: "Benchmarks",
        items: [
          { text: "LIBERO", link: "/benchmarks/libero" },
          { text: "RoboCasa", link: "/benchmarks/robocasa" },
          { text: "RoboCasa GR-1", link: "/benchmarks/robocasa-gr1" },
          { text: "RoboMimic", link: "/benchmarks/robomimic" },
          { text: "MetaWorld", link: "/benchmarks/metaworld" },
          { text: "SimplerEnv", link: "/benchmarks/simpler" },
          { text: "MS-HAB", link: "/benchmarks/mshab" },
        ],
      },
      {
        text: "Runtime Setup",
        items: [
          { text: "RoboCasa Assets", link: "/setup/robocasa-assets" },
          { text: "SimplerEnv Runtime", link: "/setup/simpler-runtime" },
          { text: "MS-HAB Runtime", link: "/setup/mshab-runtime" },
        ],
      },
      {
        text: "API",
        items: [
          { text: "Evaluate", link: "/api/evaluate" },
          { text: "Config", link: "/api/config" },
          { text: "Policies", link: "/api/policies" },
          { text: "Contracts", link: "/api/contracts" },
        ],
      },
      {
        text: "Examples",
        items: [
          {
            text: "Runnable Repository Examples",
            link: "/examples/repository-examples",
          },
          { text: "Local Random Policy", link: "/examples/local-random-policy" },
          { text: "Local Custom Policy", link: "/examples/local-custom-policy" },
          { text: "Remote Policy", link: "/examples/remote-policy" },
        ],
      },
      {
        text: "Development",
        items: [
          { text: "Adding Benchmarks", link: "/development/adding-benchmarks" },
          { text: "Contributing", link: "/contributing" },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/Chaoqi-LIU/praxis-eval" },
    ],
    search: {
      provider: "local",
    },
    footer: {
      message: "Released under the Apache-2.0 License.",
      copyright: "Copyright Chaoqi Liu and contributors.",
    },
  },
});
