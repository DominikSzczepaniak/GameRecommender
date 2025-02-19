import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { AuthContext } from '@/context/AuthProvider';
import { useContext } from 'react';
import { useLocation } from 'react-router-dom';
import { Button } from './ui/button';

const Navbar = () => {
  const { isLoggedIn } = useContext(AuthContext);
  const location = useLocation();
  const path = location.pathname;
  const excluded = path === '/login';

  const handleRedirection = (where: string) => {
    window.location.href = `/${where}`;
  };

  return (
    <div className='flex items-center m-1'>
      <Button
        className='items-center'
        onClick={() => {
          handleRedirection('');
        }}>
        <img src='../../public/logo.png' alt='Company Logo' />
      </Button>
      <div className='flex-grow flex justify-end'>
        {!isLoggedIn && !excluded && (
          <Button
            className='flex float-right'
            onClick={() => {
              handleRedirection('login');
            }}>
            <a>
              Login
            </a>
          </Button>
        )}
      </div>
      <Avatar
        className='cursor-pointer ml-1'
        onClick={() => {
          handleRedirection('settings');
        }}>
        <AvatarFallback>
          CN
        </AvatarFallback>
      </Avatar>
    </div>
  );
};

export default Navbar;
