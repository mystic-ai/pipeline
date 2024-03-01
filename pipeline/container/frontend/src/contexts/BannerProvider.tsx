import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useCallback,
} from "react";

export const BannerContext = createContext<
  | {
      showBanner: boolean;
      setBannerContent: (content: ReactNode | null) => void;
      setShowBanner: (show: boolean) => void;
      bannerContent: ReactNode | null;
    }
  | undefined
>(undefined);

export const useBannerContext = () => {
  const context = useContext(BannerContext);
  if (context === undefined) {
    throw new Error("useBannerContext must be used within a BannerProvider");
  }
  return context;
};

interface BannerProviderProps {
  children: ReactNode;
}

export const BannerProvider: React.FC<BannerProviderProps> = ({ children }) => {
  const [bannerContent, setBannerContent] = useState<ReactNode | null>(null);
  const [showBanner, setShowBanner] = useState<boolean>(false);

  // Use useCallback to memoize the setShowBanner function
  const setShowBannerCallback = useCallback((show: boolean) => {
    setShowBanner(show);
  }, []);

  return (
    <BannerContext.Provider
      value={{
        setBannerContent,
        showBanner,
        setShowBanner: setShowBannerCallback,
        bannerContent,
      }}
    >
      {children}
    </BannerContext.Provider>
  );
};
