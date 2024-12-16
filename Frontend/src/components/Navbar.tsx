import { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { useLocation } from "react-router-dom";

const Navbar = () => {
  let [logged, setLogged] = useState(false)
  let location = useLocation();
  let path = location.pathname

  let excluded = path == "/login"
  
  const handleRedirection = (where: string) => {
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
      <Button className='items-center' onClick={() => {handleRedirection("")}}>
        <img src='../../public/logo.png' alt='Company Logo' />
      </Button>
      
      <div className='flex-grow flex justify-end'>
        {(!logged && !excluded) && (
          <Button className='flex float-right' onClick={() => {handleRedirection("login")}}>
            <a>
              Login
            </a>
          </Button>
        )}
      </div>

      <Avatar className='cursor-pointer ml-1' onClick={() => {handleRedirection("settings")}}>
        <AvatarFallback>CN</AvatarFallback>
      </Avatar>
    </div>
  )
}

export default Navbar
