from datetime import datetime
import polars as pl
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class Select:
    pass



class DfMixin:
    def get_data(self):
        return self.data

    def cast_StringToFloat(self, colname:str)->pl.DataFrame:
        return self.data.with_columns(
                pl.col(colname).cast(pl.Float32).alias(colname)
                )
    
    @staticmethod
    def selectexpr(cols:List[str], datecols:List[str])->List[pl.Expr]:
        expr_list = []
        for c in cols:
            if c in datecols:
                expr_list.append(pl.col(c).str.strptime(pl.Date, fmt="%Y-%m-%d"))
            else:
                expr_list.append(pl.col(c))
        return expr_list
    
    def filter_lastrow_year(self, data, yearcol:pl.col)->pl.DataFrame:
        data = \
        data.with_columns(
            (yearcol.rank(reverse=True).over([pl.col("Id"), 
                                              pl.col("year")])\
                                                .alias("yearrank"))
            )
        data = data.filter(pl.col("yearrank") == 1).drop("yearrank")
        return data

class Patient(DfMixin):

    def __init__(self, df: pl.DataFrame) -> None:
        self.data = df

    def get_data(self):
        return self.data

    @staticmethod
    def birthdataexpr():
        return pl.col("BIRTHDATE").str.strptime(pl.Date, fmt="%Y-%m-%d")

    @staticmethod
    def load_csv(csv_path: Path):

        df = pl.read_csv(csv_path)
        df = df.select(
            Patient.selectexpr(
                ["Id", "BIRTHDATE", "DEATHDATE", "RACE", "GENDER"], 
                ["BIRTHDATE"])
        )

        return Patient(df)


class Observations(DfMixin):
    def __init__(self, df: pl.DataFrame) -> None:

        self.data = self._pivot(df)
        self._year()
        self.data = self.filter_lastrow_year(self.data,pl.col("Date"))
        self._cast()

    def _pivot(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.pivot(
            index=["Id", "Date"],
            columns="DESCRIPTION",
            values="VALUE")
        return df

    def _year(self):
        self.data = self.data.with_columns((pl.col("Date").dt.year() + 1).alias("year"))
    
    def _cast(self):
        self.data = self.cast_StringToFloat("Body Height")
        self.data = self.cast_StringToFloat("Pain severity - 0-10 verbal numeric rating [Score] - Reported")
        self.data = self.cast_StringToFloat("Body Weight")
        self.data = self.cast_StringToFloat("Body Mass Index")
        self.data = self.cast_StringToFloat("Diastolic Blood Pressure")
        self.data = self.cast_StringToFloat("Systolic Blood Pressure")
        self.data = self.cast_StringToFloat("Heart rate")
        self.data = self.cast_StringToFloat("Respiratory rate")


    def join_patient(self, patient:Patient)->None:
        self.data = self.data.join(patient.get_data(), on="Id", how="left")

    def add_age(self):
        self.data = \
            self.data.with_columns(
                    (
                        (pl.col("Date") - pl.col("BIRTHDATE") ) / (60 * 60 * 24 * 365 * 1000)
                    ).cast(pl.Int16).alias("age")
                )

    @staticmethod
    def load_csv(csv_path: Path, target_obs: List[str]):
        df = pl.read_csv(csv_path)
        df = df.filter(pl.col("DESCRIPTION").is_in(target_obs))

        df = df.select([
            (pl.col("DATE").apply(lambda x: x[0:10]).str.strptime(
                pl.Date, fmt="%Y-%m-%d").alias("Date")),
            pl.col("PATIENT").alias("Id"),
            pl.col("ENCOUNTER"),
            pl.col("DESCRIPTION"),
            pl.col("VALUE")
        ]
        )
        return Observations(df)



class Events:
    def __init__(self, df: pl.DataFrame, target_events:List[str]) -> None:

        self.data = df
        self.target_events = target_events

    def get_data(self):
        return self.data

    def get_status(self, target_year: int)->pl.DataFrame:
        evy = self.data.clone().filter(
            (pl.col("Start") < datetime(target_year, 1, 1)) &
            ((pl.col("Stop") < datetime(target_year, 1, 1))
             | (pl.col("Stop").is_null()))
        )
        evy = pl.get_dummies(evy, columns=["DESCRIPTION"])
        evy = evy.groupby("Id").agg(
            [pl.col(f"DESCRIPTION_{e}").sum().alias(e) for e in self.target_events])
        evy = evy.with_columns(
            (pl.lit(datetime(target_year, 1, 1)).dt.year()).alias("year"))
        return evy

    @staticmethod
    def load_csv(csv_path: Path, target_events: List[str]):
        df:pl.DataFrame = pl.read_csv(csv_path)
        df = df.filter(pl.col("DESCRIPTION").is_in(target_events))

        df = df.select(
            [
                pl.col("PATIENT").alias("Id"),
                pl.col("START").str.strptime(
                    pl.Date, fmt="%Y-%m-%d").alias("Start"),
                pl.col("STOP").str.strptime(
                    pl.Date, fmt="%Y-%m-%d").alias("Stop"),
                pl.col("DESCRIPTION")
            ]
        )
        return Events(df, target_events)

class SickStatus:
    def __init__(self, events:Events, target_years:List[int]):

        self.target_years = target_years
        self._create_status(events)

    def _create_status(self, events):
        st_list = []
        for y in self.target_years:
            st_list.append(events.get_status(y))
        status = pl.concat(st_list).sort(["Id", "year"])
        self.data = status

    def get_data(self)->pl.DataFrame:
        return self.data

class HealthCondition:

    def __init__(self, obs:Observations, sst:SickStatus) -> None:

        self.data = obs.get_data().join(sst.get_data(), on=["Id", "year"], how="left")
        self.data = self.data.fill_null(0)

    def get_data(self):
        return self.data

class EventSeq:

    def __init__(self, data:pl.DataFrame) -> None:
        self.rawdata = data
        self.has_patients = False

        

    def preprocess(self, maxlength):
        self.maxlength = maxlength
        data = self.rawdata.with_columns(
            pl.col("START").rank(reverse=True).over(
                pl.col("Id")
            ).alias("startrank")
        )
        data = data.filter(pl.col("startrank") <= maxlength)
        data = data.with_columns(
                            (pl.col("START").str.strptime(pl.Date, fmt="%Y-%m-%d")\
                                            .dt.year()).alias("year")
                            )
        data = data.groupby("Id").agg(
                        [pl.col("CODE").alias("events")]
                        )

        data = data.select([
            pl.col("Id").alias("Id"),
            pl.col("events")
        ])
        self.data = data

    def get_master(self):
        master = self.rawdata.groupby(["CODE", "DESCRIPTION"]).count()
        self.master = master.select([pl.col("CODE"), pl.col("DESCRIPTION")])

    def join_patient(self, patient:Patient):
        self.rawdata.join(patient, on="Id", how="left")

    @staticmethod
    def load_csv(csv_path):
        df:pl.DataFrame = pl.read_csv(csv_path)
        df = df.rename({"PATIENT": "Id"})
        return EventSeq(df)


        

