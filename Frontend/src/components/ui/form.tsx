import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { errorHandler } from '@/utilities/error'
import { useState } from "react"
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

type FormProps = React.ComponentProps<typeof Card> & {
  isLogin: boolean
}

const EMAIL_REGEX = new RegExp('^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}$')

const validator = (email: string) => {
    return EMAIL_REGEX.test(email)
}

export function LoginRegisterForm({ className, isLogin, ...props }: FormProps) {
  let [email, setEmail] = useState("")
  let [password, setPassword] = useState("")
  let [password2, setPassword2] = useState("")
  let [message, setMessage] = useState("")
  let [showPassword, setShowPassword] = useState(false)

  const fetchLoginRegister = (email: string, password: string) => {
    
  }

  const handleLoginRegister = (email: string, password: string) => {
    if (!email || !password) {
      setMessage("One of the inputs is empty!")
    }
    else if (!validator(email)) {
      setMessage("Wrong email format! Follow: example@gmail.com")
    }
    else {
      fetchLoginRegister(email, password)
      setMessage("")
    }
  }

  return (
    <Card className={cn("w-[450px]", className)} {...props}>
      <CardHeader>
        <CardTitle> {isLogin ? "Login" : "Register"}</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className=" flex items-center space-x-4 rounded-md border p-4">
          <div className="flex-1 space-y-1">
            <p>Email</p>
            <Input placeholder="Type in your email" value={email} onChange={(e) => {setEmail(e.target.value)}}/>
            <p>Password</p>
            <Input type={showPassword ? "text" : "password"} placeholder="Type in your password" value={password} onChange={(e) => {setPassword(e.target.value)}}/>
            {!isLogin && (
              <div className="mt-2">
                <p>Repeat Password</p>
                <Input className="mt-2" placeholder="Retype your password" value={password2} onChange={(e) => {setPassword2(e.target.value)}}/>
              </div>
            )}
            {isLogin && (
            <div>
              <div className="flex items-center space-x-2 text-sm text-muted-foreground mt-2">
                <Switch onClick={() => {setShowPassword(!showPassword)}}/>
                <p>Show password</p>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                <a className="mt-2 hover:text-blue-700 hover:underline transition-colors" href="/register">Not registered yet?</a>
              </p>
            </div>
            )}
            <div className="text-mm text-red">
              {message}
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button className="w-full" onClick={() => {handleLoginRegister(email, password)}}>
          {isLogin ? "Login" : "Register"}
        </Button>
      </CardFooter>
    </Card>
  )
}
