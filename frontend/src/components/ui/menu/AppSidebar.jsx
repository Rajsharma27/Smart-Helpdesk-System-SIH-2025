"use client";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Separator } from "../separator";
import { Home, Inbox, Settings } from "lucide-react";
import { useTab } from "@/context/TabContext";
import { useUser } from "@/context/UserContext";
import { LogOut } from "lucide-react";
import { attemptLogout } from "@/services/apiTicket";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";

const navigation = [
  {
    title: "My Tickets",
    value: "my-tickets",
    icon: Home,
  },
  {
    title: "Chatbot",
    value: "chatbot",
    icon: Inbox,
  },
  {
    title: "Raise a ticket",
    value: "raise-ticket",
    icon: Inbox,
  },
  {
    title: "Analytics",
    value: "analytics",
    icon: Home,
  },
  {
    title: "Settings",
    value: "settings",
    icon: Settings,
  },
];

const AppSidebar = () => {
  const { currentTab, setCurrentTab } = useTab();
  const { currentUser, isLoading, setIsLoading } = useUser();

  const router = useRouter();

  const logOut = async () => {
    setIsLoading(true);
    try {
      await attemptLogout();
      router.push("/auth");
    } catch (error) {
      console.error("Error logging out:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading || !currentUser) {
    return null;
  }

  const handleTabChange = (value) => {
    setCurrentTab(value);
  };

  return (
    <Sidebar>
      {/* Header */}
      {console.log(currentUser.user)}
      <SidebarHeader className={"flex items-center justify-center p-2"}>
        <img
          src="/assets/img/PowerGridLogo.png"
          alt="PowerGrid Logo"
          className="w-[80%]"
        />
      </SidebarHeader>
      <Separator />
      {/* Main content */}
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarGroupLabel>Navigation</SidebarGroupLabel>
            <SidebarMenu>
              {navigation.map((item) => (
                <SidebarMenuItem
                  key={item.title}
                  onClick={() => handleTabChange(item.value)}
                >
                  <SidebarMenuButton asChild>
                    <div
                      key={item.title}
                      className={cn(
                        "flex items-center gap-3 py-5 cursor-pointer",
                        currentTab === item.value &&
                          "bg-primary text-background hover:!bg-primary/90 hover:!text-background"
                      )}
                    >
                      <div>
                        <item.icon size={18} />
                      </div>
                      <span className="text-base font-extralight">
                        {item.title}
                      </span>
                    </div>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      {/* Footer */}
      <SidebarFooter
        className={"py-4 flex flex-row items-center justify-between"}
      >
        <div className="flex flex-col">
          <span className="font-semibold">{currentUser.user.name}</span>
          <span className="text-sm text-muted-foreground truncate max-w-[200px]">
            {currentUser.user.email}
          </span>
        </div>
        <div
          className="hover:cursor-pointer hover:bg-destructive/20 p-1 rounded-sm"
          onClick={logOut}
        >
          <LogOut />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
};

export default AppSidebar;
