from sqlalchemy import create_engine, Column, Integer, String, Float, Date, BigInteger, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, status
from typing import List
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

DATABASE_URL = "postgresql+psycopg2://username:password@localhost/finance_platform"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True)
    shares = Column(Integer)

    user = relationship("User", back_populates="portfolios")

User.portfolios = relationship("Portfolio", order_by=Portfolio.id, back_populates="user")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str

class UserInDB(UserCreate):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class StockPriceCreate(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class PortfolioCreate(BaseModel):
    symbol: str
    shares: int

class StockPriceResponse(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class PortfolioResponse(BaseModel):
    user_id: int
    symbol: str
    shares: int

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

Base.metadata.create_all(bind=engine)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/add_stock_price/", response_model=StockPriceResponse)
def add_stock_price(stock: StockPriceCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_stock = StockPrice(**stock.dict())
    db.add(db_stock)
    db.commit()
    db.refresh(db_stock)
    return db_stock

@app.get("/get_stock_prices/{symbol}", response_model=List[StockPriceResponse])
def get_stock_prices(symbol: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    stock_prices = db.query(StockPrice).filter(StockPrice.symbol == symbol).all()
    if not stock_prices:
        raise HTTPException(status_code=404, detail="Stock prices not found")
    return stock_prices

@app.post("/portfolios/", response_model=PortfolioResponse)
def create_portfolio(portfolio: PortfolioCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_portfolio = Portfolio(user_id=current_user.id, **portfolio.dict())
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

@app.get("/portfolios/", response_model=List[PortfolioResponse])
def get_portfolios(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    portfolios = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).all()
    return portfolios

@app.post("/moving_average/")
def calculate_moving_average(request: MovingAverageRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    symbol = request.symbol
    window_size = request.window_size
    
    stock_prices = db.query(StockPrice).filter(StockPrice.symbol == symbol).order_by(StockPrice.date).all()
    if not stock_prices:
        raise HTTPException(status_code=404, detail="Stock prices not found")
    
    df = pd.DataFrame([{
        "date": sp.date,
        "close": sp.close
    } for sp in stock_prices])
    
    df['moving_average'] = df['close'].rolling(window=window_size).mean()
    
    return JSONResponse(content=df.to_dict(orient='records'))

@app.post("/visualize_stock_prices/")
def visualize_stock_prices(symbol: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    stock_prices = db.query(StockPrice).filter(StockPrice.symbol == symbol).order_by(StockPrice.date).all()
    if not stock_prices:
        raise HTTPException(status_code=404, detail="Stock prices not found")
    
    df = pd.DataFrame([{
        "date": sp.date,
        "close": sp.close
    } for sp in stock_prices])
    
    fig = df.plot(x='date', y='close', title=f'Stock Prices for {symbol}')
    return JSONResponse(content=fig.to_json())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json

@app.post("/predict_stock_price/")
def predict_stock_price(symbol: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    stock_prices = db.query(StockPrice).filter(StockPrice.symbol == symbol).order_by(StockPrice.date).all()
    if not stock_prices:
        raise HTTPException(status_code=404, detail="Stock prices not found")
    
    df = pd.DataFrame([{
        "date": sp.date,
        "close": sp.close
    } for sp in stock_prices])
    
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].map(pd.Timestamp.to_julian_date)
    
    X = np.array(df['date']).reshape(-1, 1)
    y = np.array(df['close']).reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    results = {
        "predictions": y_pred.tolist(),
        "actual": y_test.tolist()
    }
    
    return JSONResponse(content=json.dumps(results))
