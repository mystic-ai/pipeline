import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import React from "react";
import { BannerProvider } from "./BannerProvider";
import { Notifications } from "../components/ui/Notifications/Notifications";

const TooltipProvider = TooltipPrimitive.Provider;

function DashboardProviders({ children }: React.PropsWithChildren) {
  return (
    <BannerProvider>
      <Notifications>
        <TooltipProvider>{children}</TooltipProvider>
      </Notifications>
    </BannerProvider>
  );
}

export default DashboardProviders;
