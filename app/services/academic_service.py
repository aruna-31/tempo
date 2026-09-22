from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.models.academic import Section, Student, Subject
from app.schemas.academic import (
    SectionCreate,
    SectionUpdate,
    StudentBulkCreate,
    StudentCreate,
    StudentUpdate,
    SubjectCreate,
    SubjectUpdate,
)


class AcademicService:
    # ------------------ SUBJECTS ------------------
    @staticmethod
    def create_subject(db: Session, faculty_id: UUID, data: SubjectCreate) -> Subject:
        # Check if subject code already used by this faculty
        existing = db.query(Subject).filter(
            Subject.faculty_id == faculty_id,
            Subject.code == data.code.strip()
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Subject with code '{data.code}' already exists for your account"
            )

        subject = Subject(
            faculty_id=faculty_id,
            code=data.code.strip(),
            name=data.name.strip(),
            description=data.description.strip() if data.description else None
        )
        db.add(subject)
        db.commit()
        db.refresh(subject)
        return subject

    @staticmethod
    def get_faculty_subjects(
        db: Session, faculty_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Subject]:
        return (
            db.query(Subject)
            .filter(Subject.faculty_id == faculty_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_subject_by_id(db: Session, faculty_id: UUID, subject_id: UUID) -> Subject:
        subject = (
            db.query(Subject)
            .filter(Subject.id == subject_id, Subject.faculty_id == faculty_id)
            .first()
        )
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subject not found or does not belong to your faculty account"
            )
        return subject

    @staticmethod
    def update_subject(
        db: Session, faculty_id: UUID, subject_id: UUID, data: SubjectUpdate
    ) -> Subject:
        subject = AcademicService.get_subject_by_id(db, faculty_id, subject_id)
        if data.code is not None:
            subject.code = data.code.strip()
        if data.name is not None:
            subject.name = data.name.strip()
        if data.description is not None:
            subject.description = data.description.strip()

        db.commit()
        db.refresh(subject)
        return subject

    @staticmethod
    def delete_subject(db: Session, faculty_id: UUID, subject_id: UUID) -> None:
        subject = AcademicService.get_subject_by_id(db, faculty_id, subject_id)
        db.delete(subject)
        db.commit()

    # ------------------ SECTIONS ------------------
    @staticmethod
    def create_section(db: Session, faculty_id: UUID, data: SectionCreate) -> Section:
        # Verify subject ownership
        AcademicService.get_subject_by_id(db, faculty_id, data.subject_id)

        section = Section(
            subject_id=data.subject_id,
            name=data.name.strip(),
            academic_year=data.academic_year.strip(),
            semester=data.semester.strip()
        )
        db.add(section)
        db.commit()
        db.refresh(section)
        return section

    @staticmethod
    def get_sections_by_subject(
        db: Session, faculty_id: UUID, subject_id: UUID
    ) -> List[Section]:
        # Verify ownership
        AcademicService.get_subject_by_id(db, faculty_id, subject_id)
        return db.query(Section).filter(Section.subject_id == subject_id).all()

    @staticmethod
    def get_section_by_id(db: Session, faculty_id: UUID, section_id: UUID) -> Section:
        section = (
            db.query(Section)
            .join(Subject, Section.subject_id == Subject.id)
            .filter(Section.id == section_id, Subject.faculty_id == faculty_id)
            .first()
        )
        if not section:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Section not found or does not belong to your faculty account"
            )
        return section

    @staticmethod
    def update_section(
        db: Session, faculty_id: UUID, section_id: UUID, data: SectionUpdate
    ) -> Section:
        section = AcademicService.get_section_by_id(db, faculty_id, section_id)
        if data.name is not None:
            section.name = data.name.strip()
        if data.academic_year is not None:
            section.academic_year = data.academic_year.strip()
        if data.semester is not None:
            section.semester = data.semester.strip()

        db.commit()
        db.refresh(section)
        return section

    @staticmethod
    def delete_section(db: Session, faculty_id: UUID, section_id: UUID) -> None:
        section = AcademicService.get_section_by_id(db, faculty_id, section_id)
        db.delete(section)
        db.commit()

    # ------------------ STUDENTS ------------------
    @staticmethod
    def create_student(db: Session, faculty_id: UUID, data: StudentCreate) -> Student:
        # Verify section ownership
        AcademicService.get_section_by_id(db, faculty_id, data.section_id)

        roll = data.roll_number.strip()
        existing = db.query(Student).filter(
            Student.section_id == data.section_id,
            Student.roll_number == roll
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Student with roll number '{roll}' already exists in this section"
            )

        student = Student(
            section_id=data.section_id,
            roll_number=roll,
            name=data.name.strip(),
            email=data.email.strip().lower() if data.email else None
        )
        db.add(student)
        db.commit()
        db.refresh(student)
        return student

    @staticmethod
    def bulk_create_students(
        db: Session, faculty_id: UUID, data: StudentBulkCreate
    ) -> List[Student]:
        AcademicService.get_section_by_id(db, faculty_id, data.section_id)

        created: List[Student] = []
        for item in data.students:
            roll = item.roll_number.strip()
            existing = db.query(Student).filter(
                Student.section_id == data.section_id,
                Student.roll_number == roll
            ).first()
            if not existing:
                st = Student(
                    section_id=data.section_id,
                    roll_number=roll,
                    name=item.name.strip(),
                    email=item.email.strip().lower() if item.email else None
                )
                db.add(st)
                created.append(st)

        db.commit()
        for st in created:
            db.refresh(st)
        return created

    @staticmethod
    def get_students_by_section(
        db: Session, faculty_id: UUID, section_id: UUID
    ) -> List[Student]:
        AcademicService.get_section_by_id(db, faculty_id, section_id)
        return db.query(Student).filter(Student.section_id == section_id).all()

    @staticmethod
    def get_student_by_id(db: Session, faculty_id: UUID, student_id: UUID) -> Student:
        student = (
            db.query(Student)
            .join(Section, Student.section_id == Section.id)
            .join(Subject, Section.subject_id == Subject.id)
            .filter(Student.id == student_id, Subject.faculty_id == faculty_id)
            .first()
        )
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found or not in your enrolled sections"
            )
        return student

    @staticmethod
    def update_student(
        db: Session, faculty_id: UUID, student_id: UUID, data: StudentUpdate
    ) -> Student:
        student = AcademicService.get_student_by_id(db, faculty_id, student_id)
        if data.roll_number is not None:
            student.roll_number = data.roll_number.strip()
        if data.name is not None:
            student.name = data.name.strip()
        if data.email is not None:
            student.email = data.email.strip().lower()

        db.commit()
        db.refresh(student)
        return student

    @staticmethod
    def delete_student(db: Session, faculty_id: UUID, student_id: UUID) -> None:
        student = AcademicService.get_student_by_id(db, faculty_id, student_id)
        db.delete(student)
        db.commit()
