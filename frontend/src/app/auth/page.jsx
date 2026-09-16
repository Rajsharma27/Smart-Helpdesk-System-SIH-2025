"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useUser } from "@/context/UserContext";
import { attemptLogin } from "@/services/apiTicket";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { BounceLoader } from "react-spinners";
import Link from "next/link";

const page = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const { currentUser, setCurrentUser, isLoading, setIsLoading, isLoggedIn } =
    useUser();

  const router = useRouter();

  const logIn = async () => {
    console.log("Attempting Login");
    setIsLoading(true);
    try {
      const result = await attemptLogin({ email, password });

      if (result.status !== "success") {
        throw new Error(result.msg);
      }
      // login logic
      setCurrentUser(result.data);
      // localStorage.setItem("user", JSON.stringify(result.data));
      router.push("/dashboard");
    } catch (error) {
      console.error(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isLoggedIn && currentUser && !isLoading) {
      router.push("/dashboard");
    }
  }, [isLoggedIn, currentUser, isLoading, router]);

  if (isLoading || currentUser) {
    return (
      <div className="h-screen w-screen flex items-center justify-center">
        <BounceLoader color="#4A90E2" />
      </div>
    );
  }

  return (
    <>
      {isLoading && (
        <div className="h-screen w-screen bg-black/30 backdrop-blur-sm flex items-center justify-center absolute top-0 left-0">
          <BounceLoader color="#4A90E2" />
        </div>
      )}
      <div className="grid grid-cols-2 h-screen">
        <div>
          <img
            src="/assets/img/PowerGridSunset.jpg"
            alt="PowerGrid"
            className="h-full w-full object-left object-cover"
          />
        </div>
        <div className="flex justify-center items-center">
          <Card className={"w-sm gap-10 py-8"}>
            <CardHeader>
              <CardTitle
                className={
                  "text-5xl font-extrabold text-secondary-foreground mb-1"
                }
              >
                Login
              </CardTitle>
              <CardDescription className="text-base text-muted-foreground">
                Please enter your PowerGrid credentials
              </CardDescription>
            </CardHeader>

            <CardContent className={"flex flex-col gap-6"}>
              <div>
                <Label htmlFor="email" className="mb-2">
                  Email
                </Label>
                <Input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  type="email"
                  placeholder="Email"
                  id="email"
                  className=""
                />
              </div>
              <div>
                <Label htmlFor="password" className="mb-2">
                  Password
                </Label>
                <Input
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  type="password"
                  placeholder="Password"
                  id="password"
                  className=""
                />
              </div>
            </CardContent>

            <CardFooter className="flex flex-col gap-4">
              <Button className={"w-full cursor-pointer"} onClick={logIn}>
                Login
              </Button>
              <p className="text-sm text-muted-foreground text-center">
                Don't have an account?{" "}
                <Link href="/auth/register" className="text-blue-500 hover:underline">
                  Register here
                </Link>
              </p>
            </CardFooter>
          </Card>
        </div>
      </div>
    </>
  );
};

export default page;
