export interface HeaderLink {
  href: string;
  title: string;
  target?: string;
  ariaLabel: string;
  rel?: string; // prefetch for important pages, otherwise don't use
}

export interface HeaderProductLink extends HeaderLink {
  category: string;
  description: string;
  src: string;
}
