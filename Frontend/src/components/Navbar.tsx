import React from 'react'
import { Button } from './ui/button'
import { Link } from 'lucide-react'
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

const Navbar = () => {

  return (
    <div className='flex items-center'>
      <Button className='items-center' onClick={() => {
        window.location.href = "/settings"
      }}>
        <img src='../../public/logo.png' alt='Company Logo' />
      </Button>
      
      <div className='flex-grow flex justify-end'>
        <Button className='flex float-right' onClick={() => {
        window.location.href = "/login"
      }}>
          <a>
            Login
          </a>
        </Button>
      </div>

      <Avatar className='cursor-pointer' onClick={() => {
        window.location.href = "/settings"
      }}>
        <AvatarImage src="https://github.com/shadcn.png" alt="Avatar" />
        <AvatarFallback>CN</AvatarFallback>
      </Avatar>


    </div>
  )
}

export default Navbar
