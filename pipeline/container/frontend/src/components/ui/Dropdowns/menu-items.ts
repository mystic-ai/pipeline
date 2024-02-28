import { RouteWithTitle, routes } from "@/lib/routes";

interface HeaderLink {
  href: string;
  title: string;
  target?: string;
  ariaLabel: string;
  rel?: string; // prefetch for important pages, otherwise don't use
}

interface HeaderProductLink extends HeaderLink {
  category: string;
  description: string;
  src: string;
}

export const keyRoutes: RouteWithTitle[] = [
  {
    route: routes.products.dashboard.overview,
    title: "Overview",
  },
  {
    title: "Pipelines",
    route: routes.products.dashboard.pipelines,
  },
  {
    title: "Cloud integrations",
    route: routes.products.dashboard.cloud.home,
  },
];

export const subRoutes: RouteWithTitle[] = [
  {
    title: "API Tokens",
    route: routes.products.dashboard.apiTokens,
  },
  {
    title: "Billing",
    route: routes.products.dashboard.billing,
  },
  {
    title: "Settings",
    route: routes.products.dashboard.settings,
  },
];

export const altRoutes: RouteWithTitle[] = [
  {
    title: "Documentation",
    route: routes.external.docs,
  },
  {
    title: "Community",
    route: routes.social.discord,
  },
];

export const useCaseItems: HeaderLink[] = [
  {
    href: routes.general.useCases.text.href,
    title: "Text",
    ariaLabel: routes.general.useCases.text.ariaLabel,
  },
  {
    href: routes.general.useCases.imageVideo.href,
    title: "Image and Video",
    ariaLabel: routes.general.useCases.imageVideo.ariaLabel,
  },
  {
    href: routes.general.useCases.audio.href,
    title: "Audio",
    ariaLabel: routes.general.useCases.audio.ariaLabel,
  },
  // {
  //   href: routes.products.enterprise.lander.href,
  //   title: "For enterprise",
  //   ariaLabel: routes.products.enterprise.lander.ariaLabel,
  // },
];

export const offeringsItems: HeaderLink[] = [
  {
    href: routes.general.home.href,
    title: "Serverless",
    ariaLabel: routes.general.home.ariaLabel,
  },
  {
    title: "Cloud integration",
    href: routes.products.bringYourOwnCloud.href,
    ariaLabel: routes.products.bringYourOwnCloud.ariaLabel,
  },
  {
    title: "Enterprise",
    href: routes.products.enterprise.lander.href,
    ariaLabel: routes.products.enterprise.lander.ariaLabel,
  },
];

export const resourcesItems: HeaderLink[] = [
  {
    href: routes.general.pricing.serverless.href,
    title: "Pricing",
    ariaLabel: routes.general.pricing.serverless.ariaLabel,
  },
  {
    href: routes.external.library.href,
    title: "Github",
    target: "_blank",
    rel: "noopener noreferrer",
    ariaLabel: routes.external.library.ariaLabel,
  },
  {
    href: routes.external.docs.href,
    title: "Docs",
    target: "_blank",
    rel: "noopener noreferrer",
    ariaLabel: routes.external.docs.ariaLabel,
  },
  // {
  //   href: routes.general.blog.href,
  //   title: "Blog",
  //   ariaLabel: routes.general.blog.ariaLabel,
  // },
];

export const companyItems: HeaderLink[] = [
  // {
  //   href: routes.general.about.href,
  //   title: "About",
  //   ariaLabel: routes.general.about.ariaLabel,
  // },
  {
    href: routes.social.discord.href,
    title: "Join our Discord",
    target: "_blank",
    ariaLabel: routes.social.discord.ariaLabel,
    rel: "noopener noreferrer",
  },
  // {
  //   href: routes.external.careers.href,
  //   title: "Careers",
  //   target: "_blank",
  //   ariaLabel: routes.external.careers.ariaLabel,
  //   rel: "noopener noreferrer",
  // },
  // {
  //   href: routes.general.contact.href,
  //   title: "Contact us",
  //   ariaLabel: routes.general.contact.ariaLabel,
  // },
];
