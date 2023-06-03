import React, { useState, useEffect, useCallback } from "react";
import { LoadingButton } from "@mui/lab";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "shared/api";
import { BackgroundSpinner, TextInput } from "ui";
import { authErrorMessage, validatePassword } from "shared/utilities/validators";
import { Key, LinkBreak } from "phosphor-react";

interface InvitationDetails {
  organization_name: string;
  email: string;
  invited_by_name: string;
}

export function RegistrationPage() {
  const navigate = useNavigate();
  const { token }: any = useParams();
  const [data, setData] = useState<InvitationDetails | null>();
  const [error, setError] = useState<any>();
  const [pageLoading, setPageLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  const [firstName, setFirstName] = useState("");
  const [firstNameError, setFirstNameError] = useState("");
  const [lastName, setLastName] = useState("");
  const [lastNameError, setLastNameError] = useState("");
  const [password, setPassword] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [genericError, setGenericError] = useState("");

  useEffect(() => {
    setPageLoading(true);
    async function loadData() {
      try {
        const res = await api().get(`/api/register/${token}/`);
        setData(res.data);
      } catch (err) {
        console.error(err);
        setError(err);
      } finally {
        setPageLoading(false);
      }
    }
    loadData();
  }, [token]);

  const resetValidation = () => {
    setFirstNameError("");
    setLastNameError("");
    setPasswordError("");
    setGenericError("");
  };

  const validateForm = useCallback(() => {
    resetValidation();
    let valid = true;

    if (!firstName) {
      setFirstNameError("First name is required");
      valid = false;
    }

    if (!lastName) {
      setLastNameError("Last name is required");
      valid = false;
    }

    const passwordError = authErrorMessage(validatePassword(password));
    if (passwordError) {
      setPasswordError(passwordError);
      valid = false;
    }

    return valid;
  }, [firstName, lastName, password]);

  const handleSubmit = async () => {
    if (!validateForm()) return;
    setSubmitting(true);
    try {
      const response = await api().post(`/api/register/${token}/`, {
        first_name: firstName,
        last_name: lastName,
        password: password,
      });

      if (response?.data?.success) {
        navigate("/login");
      } else {
        setGenericError(response?.data?.error_message || "Something went wrong during registration.");
      }
    } catch {
      setGenericError("Something went wrong during registration.");
    } finally {
      setSubmitting(false);
    }
  };

  const validInvitation = data && !error && !pageLoading;

  return (
    <div className="w-full md:h-screen p-4 md:p-8 mx-auto max-w-md md:max-w-6xl md:min-w-md">
      {validInvitation ? (
        <div className="h-full flex flex-col gap-4 md:flex-row items-center justify-center">
          <div className="py-8">
            <p className="pb-2 font-bold text-3xl md:text-5xl font-epilogue">Welcome to Voxel</p>
            <p className="text-gray-600">
              Your teammate <strong>{data?.invited_by_name}</strong> has invited you to join the{" "}
              <strong>{data?.organization_name}</strong> Voxel account.
            </p>
          </div>
          <div className="m-auto bg-white flex flex-col items-center justify-center p-5 rounded-xl">
            <div className="w-28 h-28 bg-brand-gray-050 flex justify-center items-center rounded-full m-4">
              <Key size={64} className="text-brand-primary-500" weight="fill" />
            </div>
            <div>
              <p className="font-bold text-2xl text-center font-epilogue">Set Your Password</p>
              <p className="text-gray-600 text-center text-md pb-4">
                Enter your name and choose a password to get started.
              </p>
            </div>
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <TextInput
                  name="firstName"
                  placeholder="First name"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  errors={[firstNameError]}
                />
                <TextInput
                  name="lastName"
                  placeholder="Last name"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  errors={[lastNameError]}
                />
              </div>
              <TextInput
                name="password"
                placeholder="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                errors={[passwordError]}
              />
              <div className="text-brand-gray-300 text-sm">
                <div className="pb-2">
                  A strong password has at least <strong>8 characters</strong>, including at least 3 of the following 4
                  types of characters:
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-2 pl-4">
                  <PasswordRequirement label="Uppercase letter" />
                  <PasswordRequirement label="Lowercase letter" />
                  <PasswordRequirement label="Number" />
                  <PasswordRequirement label="Special character (!@#$&*)" />
                </div>
              </div>
              {genericError && (
                <div className="border border-red-800 bg-red-50 text-red-800 p-4 rounded">
                  <div className="max-w-md">{genericError}</div>
                </div>
              )}
              <div className="">
                <LoadingButton
                  className="w-full"
                  variant="contained"
                  onClick={handleSubmit}
                  loading={submitting}
                  disabled={submitting}
                >
                  Set Password
                </LoadingButton>
              </div>
              <div className="text-brand-gray-300 text-sm text-center">
                By clicking "Set Password" you agree to Voxel's{" "}
                <a
                  target="_blank"
                  href="https://www.voxelai.com/terms-conditions"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  Terms & Conditions
                </a>
                .
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="px-4 flex items-center justify-center h-screen w-full">
          <div className="m-auto bg-white flex flex-col items-center justify-center p-5 rounded-xl py-10 w-96">
            <div className="w-28 h-28 flex justify-center items-center rounded-full m-4 bg-brand-gray-050/50">
              <LinkBreak size={64} color="##676785" weight="fill" />
            </div>
            <p className="font-bold text-3xl font-epilogue m-2">Invalid Invitation</p>
            <p className="text-gray-600 text-center m-2">
              It looks like this invitation link has expired. Please reach out to your administrator.
            </p>
          </div>
        </div>
      )}
      {pageLoading ? <BackgroundSpinner /> : null}
    </div>
  );
}

function PasswordRequirement(props: { label: string }) {
  return (
    <div className="flex gap-2 items-center">
      <div className="h-1 w-1 rounded-full bg-brand-gray-300">&nbsp;</div>
      <div>{props.label}</div>
    </div>
  );
}
