from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

#adding user database
user_db = {
    'jack': {'username': 'jack', 'date_joined': '2021-12-01', 'location': 'New York', 'age': 28},
    'jill': {'username': 'jill', 'date_joined': '2021-12-02', 'location': 'Los Angeles', 'age': 19},
    'jane': {'username': 'jane', 'date_joined': '2021-12-03', 'location': 'Toronto', 'age': 52}
}

class User(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    date_joined: date
    location: Optional[str] = None
    age: int = Field(None, gt=5, lt=130)  # ge: greater than or equal, le

class UserUpdate(User):
    date_joined: Optional[date] = None
    age: int = Field(None, gt=5, lt=200)

def ensure_username_in_db(username: str):
    if username not in user_db:
        raise HTTPException(status_code=404, detail=f'Username {username} not found')      



#create instance variable
app = FastAPI()



#define operations
@app.get('/users')
#define function
def get_users():
    user_list = list(user_db.values())
    return user_list

#define operations
@app.get('/users/{username}')
#define function
def get_users_path(username: str):
    return user_db[username]

@app.get('/users/{username}')
def get_users_path(username: str):
    ensure_username_in_db(username)
    return user_db[username]


#define operations
@app.get('/users')
#define function
def get_users_query(limit: int = 2):
    user_list = list(user_db.values())
    return user_list[:limit]

@app.post('/users')
def create_user(user: User):
    username = user.username
    if username in user_db:
        # status.HTTP_409_CONFLICT
        raise HTTPException(status_code=409, detail=f'Cannot create user. Username {username} already exists')
    user_db[username] = user.dict()
    return {'message': f'Successfully created user: {username}'}
