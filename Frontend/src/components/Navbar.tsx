import { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

const Navbar = () => {
  let [Logged, SetLogged] = useState(false)

  const HandleRedirection = (where: string) => {
    window.location.href = `/${where}`
  }

  useEffect(() => {
    // let user = Cookie.get("user")
    // if (user)
    // {
    //   SetLogged(true)

    // } else {
    //   SetLogged(false)
    // }
  }, [])

  return (
    <div className='flex items-center m-1'>
      <Button className='items-center' onClick={() => {HandleRedirection("")}}>
        <img src='../../public/logo.png' alt='Company Logo' />
      </Button>
      
      <div className='flex-grow flex justify-end'>
        {!Logged && (
          <Button className='flex float-right' onClick={() => {HandleRedirection("login")}}>
            <a>
              Login
            </a>
          </Button>
        )}
      </div>

      <Avatar className='cursor-pointer ml-1' onClick={() => {HandleRedirection("settings")}}>
        <AvatarFallback>CN</AvatarFallback>
      </Avatar>
    </div>
  )
}

export default Navbar
