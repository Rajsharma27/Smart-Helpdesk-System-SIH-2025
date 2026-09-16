"use client";

import { useTab } from "@/context/TabContext";
import {
  Analytics,
  Chatbot,
  MyTickets,
  RaiseTicket,
  Settings,
} from "@/components/ui/screen";
import { useEffect } from "react";
import { useUser } from "@/context/UserContext";
import { useRouter } from "next/navigation";
import { BounceLoader } from "react-spinners";

const tabs = {
  "my-tickets": MyTickets,
  chatbot: Chatbot,
  analytics: Analytics,
  "raise-ticket": RaiseTicket,
  settings: Settings,
};

const page = () => {
  const { currentTab } = useTab();
  const { isLoggedIn, isLoading, currentUser } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (!isLoggedIn && !isLoading && !currentUser) {
      router.push("/auth");
    }
  }, [isLoggedIn, isLoading, router, currentUser]);

  const CurrentTabComponent = tabs[currentTab] || MyTickets;

  if (!currentUser || isLoading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <BounceLoader color="#4A90E2" />
      </div>
    );
  }

  return (
    <div>
      <CurrentTabComponent />
    </div>
  );
};

export default page;
