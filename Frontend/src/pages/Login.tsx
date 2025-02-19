import { LoginRegisterForm } from '@/components/LoginRegisterForm';

const Login = () => {
  return (
    <div>
      <div className='flex items-center justify-center h-screen'>
        <LoginRegisterForm isLogin={true} />
      </div>
    </div>
  );
};

export default Login;
