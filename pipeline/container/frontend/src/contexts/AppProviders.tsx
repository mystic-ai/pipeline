import React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { BannerProvider } from "./BannerProvider";
import { Notifications } from "../components/ui/Notifications/Notifications";

const TooltipProvider = TooltipPrimitive.Provider;

function AppProviders({ children }: React.PropsWithChildren) {
  const [client] = React.useState(
    new QueryClient({
      defaultOptions: {
        queries: { refetchOnWindowFocus: false },
      },
    })
  );
  return (
    <>
      <QueryClientProvider client={client}>
        <BannerProvider>
          <Notifications>
            <TooltipProvider>{children}</TooltipProvider>
          </Notifications>
        </BannerProvider>
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </>
  );
}

export default AppProviders;
