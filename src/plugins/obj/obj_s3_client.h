#include <functional>
#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <string_view>
#include <stdexcept>
#include <cstdlib>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectResult.h>
#include <aws/s3/model/GetObjectResult.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/http/Scheme.h>
#include <aws/core/Aws.h>
#include "nixl_types.h"

#ifndef OBJ_S3_CLIENT_H
#define OBJ_S3_CLIENT_H

using PutObjectCallback = std::function<void (bool success)>;
using GetObjectCallback = std::function<void (bool success)>;

/**
 * Abstract interface for S3 client operations.
 * Provides async operations for PutObject and GetObject.
 */
class IS3Client {
public:
    virtual ~IS3Client() = default;

    /**
     * Set the executor for async operations.
     * @param executor The executor to use for async operations
     */
    virtual void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) = 0;

    /**
     * Asynchronously put an object to S3.
     * @param key The object key
     * @param data_ptr Pointer to the data to upload
     * @param data_len Length of the data in bytes
     * @param offset Offset within the object
     * @param callback Callback function to handle the result
     */
    virtual void
    PutObjectAsync (std::string_view key,
                    uintptr_t data_ptr,
                    size_t data_len,
                    size_t offset,
                    PutObjectCallback callback) = 0;

    /**
     * Asynchronously get an object from S3.
     * @param key The object key
     * @param data_ptr Pointer to the buffer to store the downloaded data
     * @param data_len Maximum length of data to read
     * @param offset Offset within the object to start reading from
     * @param callback Callback function to handle the result
     */
    virtual void
    GetObjectAsync (std::string_view key,
                    uintptr_t data_ptr,
                    size_t data_len,
                    size_t offset,
                    GetObjectCallback callback) = 0;
};

/**
 * Concrete implementation of IS3Client using AWS SDK S3Client.
 */
class AwsS3Client : public IS3Client {
public:
    /**
     * Constructor that creates an AWS S3Client from custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    AwsS3Client (nixl_b_params_t *custom_params,
                 std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override;

    void
    PutObjectAsync (std::string_view key,
                    uintptr_t data_ptr,
                    size_t data_len,
                    size_t offset,
                    PutObjectCallback callback) override;

    void
    GetObjectAsync (std::string_view key,
                    uintptr_t data_ptr,
                    size_t data_len,
                    size_t offset,
                    GetObjectCallback callback) override;

private:
    std::unique_ptr<Aws::SDKOptions, std::function<void (Aws::SDKOptions *)>> aws_options_;
    std::unique_ptr<Aws::S3::S3Client> s3_client_;
    Aws::String bucket_name_;
};

#endif // OBJ_S3_CLIENT_H
