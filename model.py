{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing relevant libraries \n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns \n",
    "import pickle\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(16197, 4012)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Loading data\n",
    "\n",
    "df = pd.read_csv('ELS_02_StudentFile.csv',low_memory=False)\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**DATA PREPROCESSING**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This ML model predicts college completion \n",
    "# based on a high schooler's level of financial motivation for studying\n",
    "# Data source: Education Longitudinal Study of 2002 \n",
    "\n",
    "# Target for logistic regression: received bachelor's degree\n",
    "    # F3TZBACHLTDT - Date of first known bachelor's degree \n",
    "        \n",
    "# Relevant inputs for logistic regression:\n",
    "    \n",
    "    # BYS83A - Mother's highest level of education\n",
    "    # BYS89H - Studies to increase job opportunities\n",
    "        # 1 - Almost never\n",
    "        # 2 - Sometimes\n",
    "        # 3 - Often\n",
    "        # 4 - Almost always\n",
    "    # BYS89P - Studies to ensure financial security \n",
    "        # 1 - Almost never\n",
    "        # 2 - Sometimes\n",
    "        # 3 - Often\n",
    "        # 4 - Almost always\n",
    "   \n",
    "       \n",
    "df = df.rename(columns={'STU_ID':'id','BYS83A':'mom_edu','BYS89H':'study_for_job',\n",
    "                       'BYS89P':'study_for_security','F3TZBACHLTDT':'date_firstb'})\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "8940 observations remaining\n"
     ]
    }
   ],
   "source": [
    "# Keeping features of interest\n",
    "\n",
    "data_1 = df[['id','study_for_job','mom_edu','study_for_security','date_firstb']]  \n",
    "\n",
    "\n",
    "# Dealing with missing/invalid data\n",
    "\n",
    "data_1 = data_1[ (data_1['mom_edu']>=1) & (data_1['study_for_job']>=1)\n",
    "               & (data_1['study_for_security']>=1) \n",
    "                & (data_1['date_firstb'] != -4) & (data_1['date_firstb'] != -9) ]\n",
    "        \n",
    "print(\"%d observations remaining\" % (len(data_1)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>study_for_job</th>\n",
       "      <th>mom_edu</th>\n",
       "      <th>study_for_security</th>\n",
       "      <th>bachelors</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8935</td>\n",
       "      <td>16192</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8936</td>\n",
       "      <td>16193</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8937</td>\n",
       "      <td>16194</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8938</td>\n",
       "      <td>16195</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8939</td>\n",
       "      <td>16196</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8940 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      index  study_for_job  mom_edu  study_for_security  bachelors\n",
       "0         0              1        2                   2          0\n",
       "1         1              3        5                   4          0\n",
       "2         2              2        2                   3          1\n",
       "3         3              4        2                   4          0\n",
       "4         4              3        1                   3          0\n",
       "...     ...            ...      ...                 ...        ...\n",
       "8935  16192              3        2                   3          0\n",
       "8936  16193              2        3                   2          0\n",
       "8937  16194              2        1                   3          0\n",
       "8938  16195              2        1                   3          1\n",
       "8939  16196              4        3                   4          0\n",
       "\n",
       "[8940 rows x 5 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Creating target: received bachelor's degree \n",
    "\n",
    "targets = np.where( (data_1['date_firstb']>=2007), 1, 0)\n",
    "\n",
    "data_1['bachelors'] = targets\n",
    "\n",
    "data_with_targets = data_1.drop(['date_firstb','id'],axis=1)\n",
    "data_with_targets.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Modifying feature values \n",
    "\n",
    "data_with_targets['study_for_job'] = data_with_targets['study_for_job'].map({1:0, 2:0, 3:1, 4:1})\n",
    "data_with_targets['study_for_security'] = data_with_targets['study_for_security'].map({1:0, 2:0, 3:1, 4:1})\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Train/Test Split**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train/Test Split Results:\n",
      "Train:\n",
      "8046\n",
      "Test\n",
      "894\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "data_cleaned = data_with_targets.iloc[:,:-1]\n",
    "\n",
    "# Split\n",
    "x_train, x_test, y_train, y_test = train_test_split(data_cleaned,targets,train_size=0.9,random_state=77)\n",
    "\n",
    "print(\"Train/Test Split Results:\")\n",
    "print(\"Train:\")\n",
    "print(len(x_train))\n",
    "print(\"Test\")\n",
    "print(len(x_test))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Logistic Regression with Sklearn**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "                   intercept_scaling=1, l1_ratio=None, max_iter=100,\n",
       "                   multi_class='warn', n_jobs=None, penalty='l2',\n",
       "                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,\n",
       "                   warm_start=False)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn import metrics\n",
    "\n",
    "# Training the model\n",
    "reg = LogisticRegression(solver='liblinear')\n",
    "reg.fit(x_train,y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Reg score: 0.648148\n"
     ]
    }
   ],
   "source": [
    "# Assessing Accuracy\n",
    "print(\"Reg score: %f\" % (reg.score(x_train,y_train)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Feature name</th>\n",
       "      <th>Coefficient</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>study_for_job</td>\n",
       "      <td>0.220976</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>mom_edu</td>\n",
       "      <td>0.269849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>study_for_security</td>\n",
       "      <td>0.479617</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Feature name  Coefficient\n",
       "0       study_for_job     0.220976\n",
       "1             mom_edu     0.269849\n",
       "2  study_for_security     0.479617"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Summary table\n",
    "feature_name = data_cleaned.columns.values\n",
    "summary_table = pd.DataFrame(columns=['Feature name'],data = feature_name)\n",
    "summary_table['Coefficient'] = np.transpose(reg.coef_)\n",
    "\n",
    "summary_table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Feature name</th>\n",
       "      <th>Coefficient</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Intercept</td>\n",
       "      <td>-1.904254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>study_for_job</td>\n",
       "      <td>0.220976</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>mom_edu</td>\n",
       "      <td>0.269849</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>study_for_security</td>\n",
       "      <td>0.479617</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Feature name  Coefficient\n",
       "0           Intercept    -1.904254\n",
       "1       study_for_job     0.220976\n",
       "2             mom_edu     0.269849\n",
       "3  study_for_security     0.479617"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "summary_table.index = summary_table.index + 1\n",
    "summary_table.loc[0] = ['Intercept',reg.intercept_[0]]\n",
    "summary_table = summary_table.sort_index()\n",
    "summary_table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Feature name</th>\n",
       "      <th>Coefficient</th>\n",
       "      <th>Odds ratio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Intercept</td>\n",
       "      <td>-1.904254</td>\n",
       "      <td>0.148934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>study_for_job</td>\n",
       "      <td>0.220976</td>\n",
       "      <td>1.247293</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>mom_edu</td>\n",
       "      <td>0.269849</td>\n",
       "      <td>1.309767</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>study_for_security</td>\n",
       "      <td>0.479617</td>\n",
       "      <td>1.615455</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Feature name  Coefficient  Odds ratio\n",
       "0           Intercept    -1.904254    0.148934\n",
       "1       study_for_job     0.220976    1.247293\n",
       "2             mom_edu     0.269849    1.309767\n",
       "3  study_for_security     0.479617    1.615455"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "summary_table['Odds ratio'] = np.exp(summary_table.Coefficient)\n",
    "summary_table"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Testing the Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Reg score: 0.656600\n"
     ]
    }
   ],
   "source": [
    "print(\"Reg score: %f\" % reg.score(x_test,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2    2516\n",
       "6    1986\n",
       "1    1018\n",
       "7     983\n",
       "3     879\n",
       "5     741\n",
       "4     529\n",
       "8     288\n",
       "Name: mom_edu, dtype: int64"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_cleaned['mom_edu'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Saving the Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(reg, open('model.pkl','wb'))\n",
    "\n",
    "model = pickle.load(open('model.pkl','rb'))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
