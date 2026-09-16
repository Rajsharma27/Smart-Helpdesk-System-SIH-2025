import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const colorClasses = {
  red: "text-red-500",
  blue: "text-blue-500",
  yellow: "text-yellow-500",
  green: "text-green-500",
  gray: "text-gray-500",
};

const TicketStatics = ({ label, value, icon: Icon, color }) => {
  return (
    <Card className={"gap-2 relative overflow-hidden w-full"}>
      <CardHeader className="">
        <CardTitle className="text-[24px] z-10 font-semibold font-sans text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-6xl font-bold">{value}</div>
      </CardContent>
      {Icon && (
        <Icon
          className={cn(
            "absolute right-[-24px] top-1/2 transform -translate-y-1/2 text-muted opacity-20",
            colorClasses[color] || colorClasses.gray
          )}
          size={128}
        />
      )}
    </Card>
  );
};

export default TicketStatics;
