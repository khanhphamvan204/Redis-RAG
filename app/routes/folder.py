# app/routes/folder.py
import logging
import datetime
import math
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query, Path
from pydantic import BaseModel, Field, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from typing import List, Optional, Any, Dict
from pymongo.errors import PyMongoError, DuplicateKeyError
import jwt
import bcrypt
from dotenv import load_dotenv
import os

load_dotenv()

# Import từ module services
from app.services.auth_service import verify_token_v2 

# --- 1. Cấu hình ---
MONGO_DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
AGENT_DB_NAME = "faiss_db"
FOLDER_COLLECTION = "folders"
USERS_COLLECTION = "users"
DEPARTMENTS_COLLECTION = "departments"

# Cấu hình JWT
JWT_SECRET_KEY = "678910jjj"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 6
REFRESH_TOKEN_EXPIRE_DAYS = 30 * 6

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo router
router = APIRouter()

# --- 2. Các hàm tiện ích ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Xác thực mật khẩu"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )

def get_password_hash(password: str) -> str:
    """Hash mật khẩu"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_jwt_token(data: dict, expires_delta: datetime.timedelta, token_type: str):
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.timezone.utc) + expires_delta
    to_encode.update({
        "exp": expire,
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "nbf": datetime.datetime.now(datetime.timezone.utc),
        "type": token_type,
        "jti": f"{datetime.datetime.now(datetime.timezone.utc).timestamp()}-{data.get('sub')}"
    })
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_access_token(user: dict):
    expires_delta = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": str(user["user_id"]),
        "username": user["username"],
        "user_type": user["user_type"],
        "full_name": user["full_name"],
        "user_id": user["user_id"],
        "department_id": user.get("department_id")
    }
    return create_jwt_token(payload, expires_delta, token_type="access")

def create_refresh_token(user: dict):
    expires_delta = datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = { "sub": str(user["user_id"]) }
    return create_jwt_token(payload, expires_delta, token_type="refresh")


# --- 3. Pydantic Models ---

# === Models chung ===
class DepartmentInfo(BaseModel):
    """Thông tin phòng ban rút gọn"""
    department_id: int
    department_name: str

class GenericResponse(BaseModel):
    """Response wrapper chung"""
    message: str
    status_code: int = 200
    success: bool = True
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

# === Models cho Folder ===
class FolderCreateRequest(BaseModel):
    folder_name: str = Field(..., description="Tên của folder")
    description: Optional[str] = Field(None, description="Mô tả cho folder")

class FolderInDB(BaseModel):
    folder_name: str
    description: Optional[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime

class CreateFolderResponse(BaseModel):
    message: str = "Tạo Folder thành công"
    folder: FolderInDB

class ListFolderResponse(BaseModel):
    folders: List[str]

# === Models cho User ===

# --- Model cho DB ---
class UserInDB(BaseModel):
    """Model User lưu trong MongoDB"""
    user_id: int
    username: str
    email: EmailStr
    hashed_password: str
    full_name: str
    user_type: str  # "Học sinh", "Giáo viên", "Cán bộ quản lý"
    phone_number: Optional[str] = None
    date_created: datetime.datetime
    last_login: Optional[datetime.datetime] = None
    department_id: Optional[int] = None
    student_info: Optional[Dict[str, Any]] = None
    teacher_info: Optional[Dict[str, Any]] = None
    enrollment_statistics: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

# --- Model cho Login/Register ---
class LoginRequest(BaseModel):
    """Model để đăng nhập bằng JSON"""
    username: str
    password: str

class UserRegisterRequest(BaseModel):
    """Model để tạo user mới"""
    user_id: int
    username: str
    password: str = Field(..., min_length=6)
    email: EmailStr
    full_name: str
    user_type: str
    phone_number: Optional[str] = None
    department_id: Optional[int] = None
    student_info: Optional[Dict[str, Any]] = None
    teacher_info: Optional[Dict[str, Any]] = None
    enrollment_statistics: Optional[Dict[str, Any]] = None

class UserLoginResponse(BaseModel):
    """Model Response khi đăng nhập thành công"""
    user_id: int
    username: str
    email: EmailStr
    full_name: str
    user_type: str
    phone_number: Optional[str] = None
    date_created: datetime.datetime
    last_login: Optional[datetime.datetime] = None
    
class LoginResponse(BaseModel):
    message: str = "Login successful"
    access_token: str
    refresh_token: str
    user: UserLoginResponse

# --- Models cho API /api/manager/search-user/{user_id} ---
class EnrollmentStatistics(BaseModel):
    active_enrollments: int
    completed_enrollments: int
    total_enrollments: int

class StudentInfo(BaseModel):
    date_of_birth: Optional[str] = None
    department_id: int
    enrollment_date: Optional[str] = None
    student_code: str
    student_id: int
    user_id: int

class TeacherInfo(BaseModel):
    department_id: int
    hire_date: str
    teacher_code: str
    teacher_id: int
    user_id: int

class UserDetailResponse(BaseModel):
    """Model User chi tiết cho /search-user"""
    user_id: int
    username: str
    email: EmailStr
    full_name: str
    user_type: str
    phone_number: Optional[str] = None
    date_created: datetime.datetime
    last_login: Optional[datetime.datetime] = None
    department_info: Optional[DepartmentInfo] = None
    student_info: Optional[StudentInfo] = None
    teacher_info: Optional[TeacherInfo] = None
    enrollment_statistics: Optional[EnrollmentStatistics] = None

class UserSearchData(BaseModel):
    user: UserDetailResponse

class UserSearchResponse(GenericResponse):
    data: UserSearchData
    message: str = "Tìm kiếm người dùng thành công."

# --- Models cho API /api/manager/all-users ---
class UserListInfo(BaseModel):
    """Model User rút gọn cho /all-users"""
    user_id: int
    username: str
    email: EmailStr
    full_name: str
    user_type: str
    phone_number: Optional[str] = None
    date_created: datetime.datetime
    last_login: Optional[datetime.datetime] = None
    department_info: Optional[DepartmentInfo] = None
    student_info: Optional[StudentInfo] = None
    teacher_info: Optional[TeacherInfo] = None
    
class PaginationData(BaseModel):
    page: int
    per_page: int
    total: int
    pages: int
    has_next: bool
    has_prev: bool

class FiltersApplied(BaseModel):
    department_id: Optional[int] = None
    user_type: Optional[str] = None

class AllUsersData(BaseModel):
    users: List[UserListInfo]
    pagination: PaginationData
    filters_applied: FiltersApplied

class AllUsersResponse(GenericResponse):
    data: AllUsersData
    message: str = "Lấy danh sách người dùng thành công."

# === Models cho Department ===

# --- Model cho DB ---
class DepartmentInDB(BaseModel):
    """Model Department lưu trong MongoDB"""
    department_id: int
    department_name: str
    academic_statistics: Optional[Dict[str, Any]] = None
    performance_statistics: Optional[Dict[str, Any]] = None
    personnel_statistics: Optional[Dict[str, Any]] = None
    recent_activity: Optional[Dict[str, Any]] = None
    list_statistics: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

# --- Models cho API /api/manager/departments ---
class DepartmentListStatistics(BaseModel):
    course_count: int
    student_count: int
    teacher_count: int

class DepartmentListInfo(BaseModel):
    department_id: int
    department_name: str
    statistics: Optional[DepartmentListStatistics] = None

class AllDepartmentsSummary(BaseModel):
    total_departments: int

class AllDepartmentsData(BaseModel):
    departments: List[DepartmentListInfo]
    summary: AllDepartmentsSummary

class AllDepartmentsResponse(GenericResponse):
    data: AllDepartmentsData
    message: str = "Lấy danh sách khoa thành công."

# --- Models cho API /api/manager/search-department/{department_id} ---
class AcademicStatistics(BaseModel):
    active_classes: int
    current_enrollments: int
    current_semester_classes: int
    total_courses: int
    total_credits_offered: int

class GradeDistribution(BaseModel):
    A: int = 0
    B: int = 0
    B_plus: int = Field(0, alias="B+")
    C: int = 0
    C_plus: int = Field(0, alias="C+")
    D: int = 0
    D_plus: int = Field(0, alias="D+")
    
    class Config:
        populate_by_name = True

class PerformanceStatistics(BaseModel):
    grade_distribution: GradeDistribution
    pass_rate: float
    total_graded_enrollments: int

class PersonnelStatistics(BaseModel):
    recent_students: int
    student_teacher_ratio: float
    total_students: int
    total_teachers: int

class RecentActivity(BaseModel):
    current_academic_year: str
    current_semester: str
    new_students_last_30_days: int

class DepartmentDetail(BaseModel):
    """Model Department chi tiết"""
    department_id: int
    department_name: str
    academic_statistics: Optional[AcademicStatistics] = None
    performance_statistics: Optional[PerformanceStatistics] = None
    personnel_statistics: Optional[PersonnelStatistics] = None
    recent_activity: Optional[RecentActivity] = None

class DepartmentSearchData(BaseModel):
    department: DepartmentDetail

class DepartmentSearchResponse(GenericResponse):
    data: DepartmentSearchData
    message: str = "Tìm kiếm khoa thành công."


# --- 4. Quản lý kết nối MongoDB ---
async def startup_db_client(app):
    logger.info(f"Đang kết nối đến MongoDB tại {MONGO_DATABASE_URL}...")
    try:
        app.mongodb_client = AsyncIOMotorClient(
            MONGO_DATABASE_URL,
            maxPoolSize=10, minPoolSize=2,
            serverSelectionTimeoutMS=5000
        )
        await app.mongodb_client.admin.command('ping')
        logger.info("Kết nối MongoDB thành công - Ping OK")
        app.db = app.mongodb_client[AGENT_DB_NAME]
        
        # Collection: Folders
        app.folder_collection = app.db[FOLDER_COLLECTION]
        await app.folder_collection.create_index("folder_name", unique=True)
        logger.info(f"Collection '{FOLDER_COLLECTION}' sẵn sàng.")

        # Collection: Users
        app.user_collection = app.db[USERS_COLLECTION]
        await app.user_collection.create_index("username", unique=True)
        await app.user_collection.create_index("user_id", unique=True)
        await app.user_collection.create_index("email", unique=True)
        logger.info(f"Collection '{USERS_COLLECTION}' sẵn sàng.")

        # Collection: Departments
        app.department_collection = app.db[DEPARTMENTS_COLLECTION]
        await app.department_collection.create_index("department_id", unique=True)
        logger.info(f"Collection '{DEPARTMENTS_COLLECTION}' sẵn sàng.")

    except Exception as e:
        logger.error(f"Không thể kết nối đến MongoDB: {e}")
        raise RuntimeError(f"Lỗi nghiêm trọng: Không thể kết nối DB. {e}")

async def shutdown_db_client(app):
    if hasattr(app, "mongodb_client") and app.mongodb_client:
        app.mongodb_client.close()
        logger.info("Đã đóng kết nối MongoDB.")

# --- 5. Dependencies ---
async def get_db_collection(request: Request) -> AsyncIOMotorCollection:
    if not hasattr(request.app, "folder_collection"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Dịch vụ CSDL (Folder) chưa sẵn sàng.")
    return request.app.folder_collection

async def get_user_collection(request: Request) -> AsyncIOMotorCollection:
    if not hasattr(request.app, "user_collection"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Dịch vụ CSDL (User) chưa sẵn sàng.")
    return request.app.user_collection

async def get_department_collection(request: Request) -> AsyncIOMotorCollection:
    if not hasattr(request.app, "department_collection"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Dịch vụ CSDL (Department) chưa sẵn sàng.")
    return request.app.department_collection


# --- 6. Endpoints API cho Folders ---
@router.get(
    "/agent-list/api/folders/list",
    response_model=ListFolderResponse,
    summary="Lấy danh sách tên các folder",
    tags=["Folders"]
)
async def list_folders_endpoint(collection: AsyncIOMotorCollection = Depends(get_db_collection)):
    cursor = collection.find({}, {"folder_name": 1, "_id": 0})
    folder_names = [doc["folder_name"] async for doc in cursor]
    return ListFolderResponse(folders=folder_names)

@router.post(
    "/agent-list/api/folders/add",
    response_model=CreateFolderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Tạo một folder mới",
    tags=["Folders"]
)
async def create_folder_endpoint(
    folder_data: FolderCreateRequest,
    collection: AsyncIOMotorCollection = Depends(get_db_collection),
    current_user: dict = Depends(verify_token_v2)
):
    now = datetime.datetime.now(datetime.timezone.utc)
    new_folder = FolderInDB(
        folder_name=folder_data.folder_name,
        description=folder_data.description,
        created_at=now, updated_at=now
    )
    try:
        await collection.insert_one(new_folder.model_dump())
        return CreateFolderResponse(folder=new_folder)
    except DuplicateKeyError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Folder '{folder_data.folder_name}' đã tồn tại.")
    except PyMongoError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Lỗi DB: {e}")


# --- 7. Endpoints API cho Authentication ---

@router.post(
    "/api/auth/login",
    response_model=LoginResponse,
    summary="Đăng nhập và nhận JWT",
    tags=["Authentication"]
)
async def login_for_access_token(
    login_data: LoginRequest,
    user_coll: AsyncIOMotorCollection = Depends(get_user_collection),
    dept_coll: AsyncIOMotorCollection = Depends(get_department_collection)
):
    logger.info(f"User '{login_data.username}' đang cố gắng đăng nhập.")
    
    # 1. Tìm user
    username = login_data.username.lower()
    user_doc = await user_coll.find_one({"username": username})
    
    if not user_doc or not verify_password(login_data.password, user_doc["hashed_password"]):
        logger.warning(f"Đăng nhập thất bại: Sai thông tin cho user '{username}'.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # 2. Cập nhật last_login
    now = datetime.datetime.now(datetime.timezone.utc)
    await user_coll.update_one(
        {"username": username},
        {"$set": {"last_login": now}}
    )
    user_doc["last_login"] = now
    
    # 3. Lấy thông tin department
    dept_info = None
    if user_doc.get("department_id"):
        dept_doc = await dept_coll.find_one({"department_id": user_doc["department_id"]})
        if dept_doc:
            dept_info = {
                "department_id": dept_doc["department_id"],
                "department_name": dept_doc["department_name"]
            }
    user_doc["department_info"] = dept_info
    
    # 4. Tạo tokens
    access_token = create_access_token(user_doc)
    refresh_token = create_refresh_token(user_doc)
    
    # 5. Tạo response
    user_response_data = UserLoginResponse(**user_doc)
    
    logger.info(f"User '{username}' đăng nhập thành công.")
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user_response_data
    )


# --- 8. Endpoints API cho Manager - Departments ---

@router.post(
    "/api/manager/add-department",
    summary="Thêm một phòng ban mới",
    tags=["Manager - Departments"],
    status_code=status.HTTP_201_CREATED
)
async def add_department(
    dept_data: DepartmentInDB,
    collection: AsyncIOMotorCollection = Depends(get_department_collection)
):
    """API hỗ trợ thêm dữ liệu phòng ban vào CSDL"""
    try:
        await collection.insert_one(dept_data.model_dump())
        logger.info(f"Đã thêm phòng ban: {dept_data.department_name}")
        return {"message": "Thêm phòng ban thành công", "department_id": dept_data.department_id}
    except DuplicateKeyError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Department ID '{dept_data.department_id}' đã tồn tại.")

@router.get(
    "/api/manager/departments",
    response_model=AllDepartmentsResponse,
    summary="Lấy danh sách tất cả phòng ban",
    tags=["Manager - Departments"]
)
async def get_all_departments(
    collection: AsyncIOMotorCollection = Depends(get_department_collection)
):
    departments_list = []
    cursor = collection.find({}, {
        "_id": 0,
        "department_id": 1,
        "department_name": 1,
        "list_statistics": 1 
    }).sort("department_id", 1)
    
    async for doc in cursor:
        dept_info = {
            "department_id": doc["department_id"],
            "department_name": doc["department_name"],
            "statistics": doc.get("list_statistics")
        }
        departments_list.append(DepartmentListInfo(**dept_info))
        
    data = AllDepartmentsData(
        departments=departments_list,
        summary=AllDepartmentsSummary(total_departments=len(departments_list))
    )
    return AllDepartmentsResponse(data=data)

@router.get(
    "/api/manager/search-department/{department_id}",
    response_model=DepartmentSearchResponse,
    summary="Tìm kiếm phòng ban chi tiết theo ID",
    tags=["Manager - Departments"]
)
async def search_department_by_id(
    department_id: int = Path(..., description="ID của phòng ban cần tìm"),
    collection: AsyncIOMotorCollection = Depends(get_department_collection)
):
    dept_doc = await collection.find_one(
        {"department_id": department_id},
        {"_id": 0}
    )
    
    if not dept_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy phòng ban với ID {department_id}"
        )
        
    dept_detail = DepartmentDetail(**dept_doc)
    
    return DepartmentSearchResponse(
        data=DepartmentSearchData(department=dept_detail)
    )


# --- 9. Endpoints API cho Manager - Users ---

@router.post(
    "/api/manager/add-user",
    summary="Thêm một user mới",
    tags=["Manager - Users"],
    status_code=status.HTTP_201_CREATED
)
async def add_complex_user(
    user_data: UserRegisterRequest,
    collection: AsyncIOMotorCollection = Depends(get_user_collection)
):
    """API hỗ trợ thêm user với đầy đủ thông tin"""
    hashed_password = get_password_hash(user_data.password)
    now = datetime.datetime.now(datetime.timezone.utc)
    
    new_user = UserInDB(
        user_id=user_data.user_id,
        username=user_data.username.lower(),
        email=user_data.email.lower(),
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        user_type=user_data.user_type,
        phone_number=user_data.phone_number,
        date_created=now,
        department_id=user_data.department_id,
        student_info=user_data.student_info,
        teacher_info=user_data.teacher_info,
        enrollment_statistics=user_data.enrollment_statistics
    )
    
    try:
        await collection.insert_one(new_user.model_dump())
        logger.info(f"Đã thêm user: {new_user.username}")
        return {"message": "Thêm user thành công", "user_id": new_user.user_id}
    except DuplicateKeyError as e:
        detail = f"Lỗi trùng lặp: {e.details.get('keyPattern', {})}"
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)

@router.get(
    "/api/manager/all-users",
    response_model=AllUsersResponse,
    summary="Lấy danh sách người dùng (phân trang, lọc)",
    tags=["Manager - Users"]
)
async def get_all_users(
    collection: AsyncIOMotorCollection = Depends(get_user_collection),
    dept_coll: AsyncIOMotorCollection = Depends(get_department_collection),
    page: int = Query(1, ge=1, description="Số trang"),
    per_page: int = Query(10, ge=1, le=100, description="Số mục mỗi trang"),
    department_id: Optional[int] = Query(None, description="Lọc theo ID phòng ban"),
    user_type: Optional[str] = Query(None, description="Lọc theo loại người dùng")
):
    skip = (page - 1) * per_page
    
    # Xây dựng filter
    match_filter = {}
    if department_id:
        match_filter["department_id"] = department_id
    if user_type:
        match_filter["user_type"] = user_type
    
    # Đếm tổng số
    total_users = await collection.count_documents(match_filter)
    
    # Lấy dữ liệu
    cursor = collection.find(
        match_filter,
        {"_id": 0, "hashed_password": 0}
    ).sort("user_id", 1).skip(skip).limit(per_page)
    
    users_list = []
    async for user_doc in cursor:
        # Lấy thông tin department nếu có
        dept_info = None
        if user_doc.get("department_id"):
            dept_doc = await dept_coll.find_one(
                {"department_id": user_doc["department_id"]},
                {"_id": 0, "department_id": 1, "department_name": 1}
            )
            if dept_doc:
                dept_info = DepartmentInfo(**dept_doc)
        
        user_doc["department_info"] = dept_info
        users_list.append(UserListInfo(**user_doc))
    
    total_pages = math.ceil(total_users / per_page)
    
    pagination = PaginationData(
        page=page,
        per_page=per_page,
        total=total_users,
        pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
    
    filters = FiltersApplied(department_id=department_id, user_type=user_type)
    
    data = AllUsersData(
        users=users_list,
        pagination=pagination,
        filters_applied=filters
    )
    
    return AllUsersResponse(data=data)

@router.get(
    "/api/manager/search-user/{user_id}",
    response_model=UserSearchResponse,
    summary="Tìm kiếm người dùng chi tiết theo ID",
    tags=["Manager - Users"]
)
async def search_user_by_id(
    user_id: int = Path(..., description="ID của người dùng cần tìm"),
    user_coll: AsyncIOMotorCollection = Depends(get_user_collection),
    dept_coll: AsyncIOMotorCollection = Depends(get_department_collection)
):
    # 1. Tìm user
    user_doc = await user_coll.find_one({"user_id": user_id}, {"_id": 0, "hashed_password": 0})
    
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy người dùng với ID {user_id}"
        )
        
    # 2. Lấy thông tin department
    dept_info = None
    if user_doc.get("department_id"):
        dept_doc = await dept_coll.find_one(
            {"department_id": user_doc["department_id"]},
            {"_id": 0, "department_id": 1, "department_name": 1}
        )
        if dept_doc:
            dept_info = DepartmentInfo(**dept_doc)
            
    user_doc["department_info"] = dept_info
    
    # 3. Parse bằng Pydantic
    try:
        user_detail = UserDetailResponse(**user_doc)
    except Exception as e:
        logger.error(f"Lỗi Pydantic parsing cho user {user_id}: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Lỗi định dạng dữ liệu")

    return UserSearchResponse(
        data=UserSearchData(user=user_detail)
    )