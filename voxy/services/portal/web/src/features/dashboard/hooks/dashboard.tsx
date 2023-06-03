import { createContext, useContext, useState } from "react";

interface DashboardContextInterface {
  feedSlideOpen: boolean;
  setFeedSlideOpen: (value: boolean) => void;
}

const stub = (): never => {
  throw new Error("You forgot to wrap your component in <DashboardProvider>.");
};

const initialContext = {
  feedSlideOpen: false,
  setFeedSlideOpen: stub,
};

const DashboardContext = createContext<DashboardContextInterface>(initialContext);

export const DashboardProvider: React.FC<React.ReactNode> = ({ children }) => {
  const [feedSlideOpen, setFeedSlideOpen] = useState<boolean>(false);

  return (
    <DashboardContext.Provider
      value={{
        feedSlideOpen,
        setFeedSlideOpen,
      }}
    >
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboardContext = () => {
  return useContext(DashboardContext);
};
