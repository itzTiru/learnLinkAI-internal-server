from pydantic import BaseModel, EmailStr

class UserSignup(BaseModel):
    client_name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str
